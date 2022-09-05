//! A lock-free, fixed size queue.
//!
//! Principle of operation:
//!
//! The queue is implemented as a fixed-size array of atomic `SeqPointer` types.
//!
//! A `SeqPointer` can contain either a pointer or a sequence number. A sequence
//! number means that slot in the array is empty.
//!
//! Pushing to the queue is accomplished by atomically writing a new T pointer
//! to the head of the queue, then incrementing the head index.
//!
//! Popping a value from the queue is accomplished by incrementing the tail
//! index, and then atomically exchanging a placeholder value for the pointer
//! that was stored at that index.
//!
//! How a push works, in more detail:
//!
//!   1. Read the head.
//!   2. Read the tail.
//!   3. If we're not on our first iteration, check whether the head is
//!      unchanged since the previous iteration. If so, check if the tail
//!      is exactly one sequence number behind the head.
//!      a. If it is, the queue is full, and we return an Error.
//!      b. Otherwise, another thread wrote a pointer to the array, but
//!         stalled before incrementing the head. Since all head increments
//!         are equivalent, we will try to increment the head. If that swap
//!         succeeds, keep going. Otherwise start over.
//!   4. Try to atomically swap our T pointer into the head location of the
//!      array (from sequence-number to pointer). If that swap fails, start
//!      over.
//!   5. Try to atomically swap an updated head. If that swap fails,
//!      another sender incremented the head before us. Their write
//!      accomplishes the same thing as ours, so either way we can exit
//!      sucessfully.
//!
//! How a pop works, in more detail:
//!
//!   1. Read the head.
//!   2. Read the tail.
//!   3. If the head and the tail are the same, the queue is empty.
//!      (Note wraparound to full looks different: head will be at seq N+1.)
//!   4. We know that there were some values in the queue, so it's safe to move the tail.
//!   5. Atomically increment the tail. If that fails, go back to 1.
//!   6. We now "own" that slot; swap a new sequence number into that slot,
//!      and receive its contents (which must be a pointer).
//!      If we got a seq rather than a pointer, that's a bug.
//!
//! TODO:
//! - For items that are smaller than a pointer, allow by-value storage.
//!

use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

struct AtomicSeqPointer<T> {
    inner: AtomicPtr<T>,
    phantom: PhantomData<T>,
}

// impl by hand so we don't get a trait bound `where T: Default`.
impl<T> Default for AtomicSeqPointer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AtomicSeqPointer<T> {
    const fn new() -> Self {
        let val = SeqPointer::<T>::new();
        Self {
            inner: AtomicPtr::new(val.into_raw()),
            phantom: PhantomData,
        }
    }

    fn read(&self) -> SeqPointer<T> {
        let val = self.inner.load(Ordering::Acquire);
        SeqPointer::from_raw(val)
    }

    fn _write(&self, val: SeqPointer<T>) {
        let val = val.into_raw();
        self.inner.store(val, Ordering::Release);
    }

    /// Returns true if the update succeeded.
    ///
    fn try_update(&self, old: SeqPointer<T>, new: SeqPointer<T>) -> bool {
        let old = old.into_raw();
        let new = new.into_raw();
        self.inner
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn swap(&self, new: SeqPointer<T>) -> SeqPointer<T> {
        let new = new.into_raw();
        let old = self.inner.swap(new, Ordering::AcqRel);
        SeqPointer::from_raw(old)
    }
}

/// A `SeqPointer` holds either a pointer or a sequence number.
///
/// There are two variants possible:
/// - Pointer: stored exactly like a regular pointer.
/// - Sequence Number: least-significant bit is 1. Other bits contain the count.
///
struct SeqPointer<T> {
    inner: *mut T,
    phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for SeqPointer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SeqPointer").field(&self.inner).finish()
    }
}

// impl by hand so we don't get a trait bound `where T: Default`.
impl<T> Default for SeqPointer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SeqPointer<T> {
    const fn new() -> Self {
        Self {
            // 1 is the tag for a sequence number.
            inner: sptr::invalid_mut(1),
            phantom: PhantomData,
        }
    }

    fn from_raw(raw: *mut T) -> Self {
        Self {
            inner: raw,
            phantom: PhantomData,
        }
    }

    fn from_pointer(pointer: *mut T) -> Self {
        Self {
            inner: pointer,
            phantom: PhantomData,
        }
    }

    fn from_seq(count: usize) -> Self {
        // FIXME: detect overflow during shift?
        let count = count << 1 | 1;
        Self {
            inner: sptr::invalid_mut(count),
            phantom: PhantomData,
        }
    }

    fn is_pointer(&self) -> bool {
        #![allow(unstable_name_collisions)]
        use sptr::Strict;
        self.inner.addr() & 0x1 == 0
    }

    fn _is_seq(&self) -> bool {
        #![allow(unstable_name_collisions)]
        use sptr::Strict;
        self.inner.addr() & 0x1 != 0
    }

    const fn into_raw(self) -> *mut T {
        self.inner
    }

    fn into_pointer(self) -> *mut T {
        assert!(self.is_pointer());
        self.inner as *mut T
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
struct SeqIndex<const N: usize>(usize);

impl<const N: usize> SeqIndex<N> {
    fn _from_seq_index(seq: usize, index: usize) -> Self {
        SeqIndex(seq * N + index)
    }

    /// Split the SeqIndex into a sequence number and index.
    fn split(&self) -> (usize, usize) {
        let seq = self.0 / N;
        let index = self.0 % N;
        (seq, index)
    }

    fn increment(&self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    fn increment_seq(&self) -> Self {
        Self(self.0.wrapping_add(N))
    }
}

#[derive(Default)]
struct AtomicSeqIndex<const N: usize> {
    inner: AtomicUsize,
}

impl<const N: usize> AtomicSeqIndex<N> {
    const fn new() -> Self {
        Self {
            inner: AtomicUsize::new(0),
        }
    }

    fn read(&self) -> SeqIndex<N> {
        let val = self.inner.load(Ordering::Acquire);
        SeqIndex(val)
    }

    /// Returns true if the update was successful.
    fn try_update(&self, old: SeqIndex<N>, new: SeqIndex<N>) -> bool {
        let old = old.0;
        let new = new.0;
        self.inner
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// A `push` failed because the queue is full.
#[derive(Debug)]
pub struct QueueFull<T>(pub T);

/// A lock-free concurrent queue with fixed size
///
/// # Examples
/// ```
/// # use std::sync::Arc;
/// use std::thread;
/// # use std::time::Duration;
/// use fixed_channel::queue::FixedQueue;
///
/// let queue: Arc<FixedQueue<u32, 128>> = Arc::default();
///
/// let mut threads: Vec<_> = (0..16)
/// .map(|ii| {
///     let queue = Arc::clone(&queue);
///     thread::spawn(move || {
///         // Push one number, then pop one and return it.
///         queue.push(Box::new(ii)).expect("push failed");
///         thread::yield_now();
///         *queue.pop().expect("pop failed")
///     })
/// })
/// .collect();
///
/// let sum: u32 = threads.drain(..).map(|t| t.join().unwrap()).sum();
/// assert_eq!(sum, 120);
/// ```
pub struct FixedQueue<T, const N: usize> {
    pointer_array: Box<[AtomicSeqPointer<T>]>,
    phantom: PhantomData<T>,
    head: AtomicSeqIndex<N>,
    tail: AtomicSeqIndex<N>,
}

impl<T, const N: usize> Default for FixedQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for FixedQueue<T, N> {
    fn drop(&mut self) {
        for slot in &*self.pointer_array {
            let val_maybe = slot.read();
            if val_maybe.is_pointer() {
                let raw = val_maybe.into_pointer();
                // SAFETY: todo
                unsafe {
                    let _ = Box::from_raw(raw);
                }
            }
        }
    }
}

impl<T, const N: usize> FixedQueue<T, N> {
    /// Create a new empty `FixedQueue`.
    ///
    // FIXME: make this const
    pub fn new() -> Self {
        let mut elements = Vec::new();
        elements.resize_with(N, || AtomicSeqPointer::<T>::new());
        Self {
            pointer_array: elements.into_boxed_slice(),
            phantom: PhantomData,
            head: AtomicSeqIndex::new(),
            tail: AtomicSeqIndex::new(),
        }
    }

    /// Try to push a value into the queue.
    pub fn push(&self, val: Box<T>) -> Result<(), QueueFull<Box<T>>> {
        let raw = Box::leak(val);
        let mut prev_head = None;
        loop {
            // Must read first the head, then the tail. If that's reversed, then
            // we could incorrectly get an old tail and a new head, and if
            // they're exactly the right distance apart in time, we may falsely
            // report that the queue is full. If we get an old head and a new
            // tail, we won't conclude anything incorrect and will try again.
            //
            let mut head = self.head.read();
            let tail = self.tail.read();
            if let Some(prev) = prev_head {
                if prev == head {
                    // We previously attempted to write to this exact head slot and
                    // failed, which means one of two things happened:
                    // 1. The queue is full.
                    // 2. Another thread stalled after writing the pointer but before
                    //    updating the head.
                    //
                    // Try to detect whether the queue is full. Both the head and tail
                    // might be in motion, so comparing them is somewhat fraught. But
                    // eventually we will either detect a full queue, or we will make
                    // forward progress.
                    if tail.increment_seq() == head {
                        // The queue is full.
                        //
                        // SAFETY: this function has exclusive ownership of the `raw` pointer,
                        // which we leaked ourselves at the top of this function and have
                        // failed to write into the queue. Therefore reconstituting the Box
                        // is always sound.
                        let reboxed = unsafe { Box::from_raw(raw) };
                        return Err(QueueFull(reboxed));
                    }

                    // Attempt to increment the head on behalf of the thread that is stalled.
                    let new_head = head.increment();
                    if self.head.try_update(head, new_head) {
                        // We updated the head, so perhaps we can make progress now.
                        head = new_head;
                    } else {
                        // We failed to update the head, so we are racing with other
                        // threads (that are making progress), so we need to retry.
                        prev_head = Some(head);
                        continue;
                    }
                }
            }
            let (seq, index) = head.split();
            let old = SeqPointer::from_seq(seq);
            let new = SeqPointer::from_pointer(raw);
            if self.pointer_array[index].try_update(old, new) {
                // success!
                self.head.try_update(head, head.increment());
                return Ok(());
            }
            // We failed to update the head.
            // Go back and start over.
            prev_head = Some(head);
        }
    }

    /// Attempt to pop a new value from the queue.
    pub fn pop(&self) -> Option<Box<T>> {
        loop {
            // FIXME: does the order of these two matter?
            let head = self.head.read();
            let tail = self.tail.read();
            if head == tail {
                // Queue is empty
                return None;
            }
            // We know that there were a nonzero number of items in the queue,
            // at the moment we read the tail. Therefore it's safe to attempt
            // to move the tail.
            let new_tail = tail.increment();
            if self.tail.try_update(tail, new_tail) {
                // We successfully moved the tail, so whatever value is at the
                // old tail index is ours exclusively.
                let (seq, index) = tail.split();
                let new_seq = SeqPointer::from_seq(seq + 1);
                let val = self.pointer_array[index].swap(new_seq);
                // It shouldn't be possible to hit a sequence number here,
                // since we are the exclusive owner of this slot.
                let raw = val.into_pointer();

                // SAFETY: todo
                let boxed_val = unsafe { Box::from_raw(raw) };
                return Some(boxed_val);
            }
        }
        // We failed to move the tail, so we're racing with someone and must
        // start over.
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn basic_test() {
        let queue = FixedQueue::<u32, 128>::new();
        queue.push(Box::new(42)).unwrap();
        queue.push(Box::new(43)).unwrap();

        assert_eq!(*queue.pop().unwrap(), 42);
        assert_eq!(*queue.pop().unwrap(), 43);
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn fill_then_empty() {
        let queue = FixedQueue::<u32, 4>::new();

        for ii in 0u32..4 {
            // First loop will write 1,2,3. Second loop 11, 12, 13...
            let ii = ii * 10;
            queue.push(Box::new(ii + 1)).unwrap();
            queue.push(Box::new(ii + 2)).unwrap();
            queue.push(Box::new(ii + 3)).unwrap();
            queue.push(Box::new(ii + 4)).unwrap();
            queue.push(Box::new(ii + 5)).unwrap_err();
            assert_eq!(*queue.pop().unwrap(), ii + 1);
            assert_eq!(*queue.pop().unwrap(), ii + 2);
            assert_eq!(*queue.pop().unwrap(), ii + 3);
            assert_eq!(*queue.pop().unwrap(), ii + 4);
            assert_eq!(queue.pop(), None);
        }
    }

    #[test]
    fn racing_readers() {
        #[cfg(miri)]
        const SIZE: usize = 1024;
        #[cfg(not(miri))]
        const SIZE: usize = 128 * 1024;
        let queue: Arc<FixedQueue<u32, SIZE>> = Arc::default();
        for ii in 0u32..(SIZE as u32) {
            queue.push(Box::new(ii)).unwrap();
        }
        let mut threads: Vec<_> = (0..16)
            .map(|_| {
                let queue = Arc::clone(&queue);
                std::thread::spawn(move || {
                    let mut sum = 0u64;
                    while let Some(n) = queue.pop() {
                        sum += *n as u64;
                    }
                    sum
                })
            })
            .collect();

        let sum: u64 = threads.drain(..).map(|t| t.join().unwrap()).sum();

        let expected_sum = SIZE * (SIZE - 1) / 2;
        assert_eq!(sum, expected_sum as u64);
    }

    #[test]
    fn push_pop() {
        use std::sync::Arc;
        use FixedQueue;

        const SIZE: usize = 1024;
        const THREADS: u32 = 64;
        let queue: Arc<FixedQueue<u32, SIZE>> = Arc::default();
        let mut threads: Vec<_> = (0..THREADS)
            .map(|ii| {
                let queue = Arc::clone(&queue);
                std::thread::spawn(move || {
                    // Push one number, then pop one and return it.
                    queue.push(Box::new(ii)).unwrap();
                    std::thread::yield_now();
                    *queue.pop().unwrap()
                })
            })
            .collect();

        let sum: u32 = threads.drain(..).map(|t| t.join().unwrap()).sum();
        assert_eq!(sum, THREADS * (THREADS - 1) / 2);
    }

    // To run:  cargo test run_tests_forever -- --ignored
    #[ignore = "loops forever"]
    #[test]
    fn run_tests_forever() {
        loop {
            fill_then_empty();
            racing_readers();
            push_pop();
        }
    }
}
