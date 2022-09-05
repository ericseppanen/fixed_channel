//! Principle of operation:
//!
//! The queue is implemented as a fixed-size array of atomic `SeqPointer` types.
//!
//! A `SeqPointer` can contain either a pointer or a sequence number. A sequence
//! number means that slot in the array is empty.
//!
//! Pushing to the queue is accomplished by writing a new T pointer to the head
//! of the queue, then incrementing the tail index.
//!
//! Popping a value from the queue is accomplished by atomically swapping an
//! "empty" `SeqPointer` with the tail of the queue, receiving in exchange
//! the pointer that was there.
//!
//! How a send works, in more detail:
//!
//!   1. Read the head_data (head index and sequence number).
//!   2. Try to atomically swap our T pointer into that head location
//!      (from empty to pointer)
//!      a. if that swap fails, go back to 1
//!         (optional#1: or maybe try the next location?)
//!      b. if the swap succeeds, we have successfully pushed. Keep going.
//!   3. Try to atomically swap an updated head_data.
//!      a. if that swap fails, optional#1 must be in play? A sender that
//!         came after us wrote the head before us. Their write supercedes
//!         ours, so we can exit sucessfully.
//!      b. if that swap succeeds, we can exit successfully.
//!
//! How a receive works, in more detail:
//!
//!   1. Read the head.
//!   2. Read the tail.
//!   3. If the head and the tail are the same, the queue is empty.
//!      (Note wraparound to full looks different: head will be at seq N+1.)
//!   4. We know that there were some values in the queue, so it's safe to move the tail.
//!   5. Atomically increment the tail. If that fails, go back to 1.
//!   6. We now "own" that slot; swap a new sequence number into that slot,
//!      and receive back its contents (which must be a pointer).
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

    // Returns true if the update was successful.
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
            let head = self.head.read();
            if let Some(prev) = prev_head {
                if prev == head {
                    // Failed to push, and not racing with anyone else.
                    // Must be full.
                    // SAFETY: todo
                    let reboxed = unsafe { Box::from_raw(raw) };
                    return Err(QueueFull(reboxed));
                }
            }
            let (seq, index) = head.split();
            let old = SeqPointer::from_seq(seq);
            let new = SeqPointer::from_pointer(raw);
            if self.pointer_array[index].try_update(old, new) {
                // success!
                let up = self.head.try_update(head, head.increment());
                // We're not doing the optional#1 trick yet, so this always works.
                assert!(up);
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
}