//! Principle of operation:
//!
//! The channel is implemented as a fixed-size array of atomic "pointers".
//! The pointers are special in that they may store several types of data:
//! - A pointer to a T
//! - Special values that mean "this slot is empty"
//!
//! - A send is accomplished by writing a new T pointer to the tail of the
//!   queue, then incrementing the tail index.
//! - A receive is accomplished by atomically swapping an "empty" token into
//!   the head of the queue, and receiving the T pointer that was there.
//!
//! - A send goes like this:
//!   1. Read the head_data (head index and sequence number).
//!   2. Try to atomically swap our T pointer into that head location
//!      (from seq-null to t)
//!   2.a. if that swap fails, go back to 1
//!        (optional#1: or maybe try the next location?)
//!   2.b. if the swap succeeds, we have successfully pushed. Keep going.
//!   3. Try to atomically swap an updated head_data.
//!   3.a. if that swap fails, optional#1 must be in play? A sender that
//!        came after us wrote the head before us. Their write supercedes
//!        ours, so we can exit sucessfully.
//!   3.b. if that swap succeeds, we can exit successfully.
//!
//!   How does a sender know to not spin forever when the head catches the tail?
//!   If step 2 fails but step 1 reads the same head_data, then a try_send would
//!   fail.
//!
//!   How does a sender wake a sleeping receiver? Should we wake additional receivers
//!   each time a send succeeds?
//!
//! - A receive goes like this:
//!   1. Read the tail_data (tail index and sequence number).
//!   2. Atomically read the tail.
//!   3. If the value is empty, re-read the tail_data. If it hasn't changed, we
//!      must sleep (recv) or return None (try_recv).
//!   4. Try to increment the tail data (atomic swap from N to N+1)
//!   4.a. if that swap fails, someone else read this value; go back to 1.
//!        (optional#2: or maybe try the next location?)
//!   4.b. if that swap succeeds, the data is ours. Exit successfully.
//!
//! TODO:
//! - For items that are smaller than a pointer, allow by-value storage.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

struct AtomicSeqPointer<T> {
    inner: AtomicUsize,
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
            inner: AtomicUsize::new(val.into_raw()),
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
}

/// A `SeqPointer` holds either a pointer or a sequence number.
///
/// There are two variants possible:
/// - Pointer: stored exactly like a regular pointer.
/// - Sequence Number: least-significant bit is 1. Other bits contain the count.
///
struct SeqPointer<T> {
    inner: usize,
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
            inner: 1,
            phantom: PhantomData,
        }
    }

    fn from_raw(raw: usize) -> Self {
        Self {
            inner: raw,
            phantom: PhantomData,
        }
    }

    fn from_pointer(pointer: *mut T) -> Self {
        Self {
            inner: pointer as usize,
            phantom: PhantomData,
        }
    }

    fn from_seq(count: usize) -> Self {
        // FIXME: detect overflow during shift?
        let count = count << 1 | 1;
        Self {
            inner: count,
            phantom: PhantomData,
        }
    }

    fn is_pointer(&self) -> bool {
        self.inner & 0x1 == 0
    }

    fn is_seq(&self) -> bool {
        self.inner & 0x1 != 0
    }

    const fn into_raw(self) -> usize {
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
        Self(self.0 + 1)
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

pub struct FixedChannel<T, const N: usize> {
    pointer_array: [AtomicSeqPointer<T>; N],
    phantom: PhantomData<T>,
    head: AtomicSeqIndex<N>,
    tail: AtomicSeqIndex<N>,
}

impl<T, const N: usize> Default for FixedChannel<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> FixedChannel<T, N> {
    // FIXME: make this const
    pub fn new() -> Self {
        let pointer_array = core::array::from_fn(|_| AtomicSeqPointer::new());
        Self {
            pointer_array,
            phantom: PhantomData,
            head: AtomicSeqIndex::new(),
            tail: AtomicSeqIndex::new(),
        }
    }

    // - A send goes like this:
    //   1. Read the head_data (head index and sequence number).
    //   2. Try to atomically swap our T pointer into that head location
    //      (from seq-null to t)
    //   2.a. if that swap fails, go back to 1
    //        (optional#1: or maybe try the next location?)
    //   2.b. if the swap succeeds, we have successfully pushed. Keep going.
    //   3. Try to atomically swap an updated head_data.
    //   3.a. if that swap fails, optional#1 must be in play? A sender that
    //        came after us wrote the head before us. Their write supercedes
    //        ours, so we can exit sucessfully.
    //   3.b. if that swap succeeds, we can exit successfully.
    pub fn push(&self, val: Box<T>) {
        let raw = Box::leak(val);
        loop {
            let head_info = self.head.read();
            let (seq, index) = head_info.split();
            let old = SeqPointer::from_seq(seq);
            let new = SeqPointer::from_pointer(raw);
            if self.pointer_array[index].try_update(old, new) {
                // success!
                let up = self.head.try_update(head_info, head_info.increment());
                // We're not doing the optional#1 trick yet, so this always works.
                assert!(up);
                return;
            }
            // We failed to update the head.
            // Go back and start over.
        }
    }

    // - A receive goes like this:
    //   1. Read the tail_data (tail index and sequence number).
    //   2. Atomically read the tail.
    //   3. If the value is empty, re-read the tail_data. If it hasn't changed, we
    //      must sleep (recv) or return None (try_recv).
    //   4. Try to increment the tail data (atomic swap from N to N+1)
    //   4.a. if that swap fails, someone else read this value; go back to 1.
    //        (optional#2: or maybe try the next location?)
    //   4.b. if that swap succeeds, the data is ours. Exit successfully.
    //
    pub fn pop(&self) -> Option<Box<T>> {
        loop {
            // Read the tail data.
            let mut tail = self.tail.read();
            let val_maybe = loop {
                let (_, index) = tail.split();
                let val_maybe = self.pointer_array[index].read();
                if val_maybe.is_pointer() {
                    // A value is present at this location; we can move forward.
                    break val_maybe;
                }
                if val_maybe.is_seq() {
                    // There is no value at the tail. Need to re-read the tail info.
                    let prev_tail = tail;
                    tail = self.tail.read();
                    if tail == prev_tail {
                        // No change; the queue is empty.
                        return None;
                    }
                }
                // Someone else changed the tail, so we need to start again.
            };
            let new_tail = tail.increment();
            if self.tail.try_update(tail, new_tail) {
                let raw = val_maybe.into_pointer();

                // SAFETY: todo
                let boxed_val = unsafe { Box::from_raw(raw) };
                return Some(boxed_val);
            }
            // If the update failed, someone else grabbed this
            // value. We need to start over.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        let channel = FixedChannel::<u32, 128>::new();
        channel.push(Box::new(42));
        channel.push(Box::new(43));

        assert_eq!(channel.pop().as_deref(), Some(&42));
        assert_eq!(channel.pop().as_deref(), Some(&43));
        assert_eq!(channel.pop(), None);
    }
}
