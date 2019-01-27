#ifndef MYTHREAD_H
#define MYTHREAD_H

#include <stddef.h>

/// This is the type of a function run in a mythread thread
typedef void (*mythread_fn_t)();

/// A thread handle will just be an index into our global array of threads
typedef int mythread_t;

/**
 * Initialize the mythread library. Programs should call this before calling
 * any other mythread functions.
 */
void mythread_init();

/**
 * Create a new thread.
 *
 * \param handle  The handle for this thread will be written to this location.
 * \param fn      The new thread will run this function.
 */
void mythread_create(mythread_t* handle, mythread_fn_t fn);

/**
 * Join with a thread.
 *
 * \param handle  This is the handle produced by mythread_create.
 */
void mythread_join(mythread_t handle);

/**
 * The currently-executing thread should sleep and yield the CPU.
 * 
 * \param ms  The number of milliseconds the thread should sleep.
 */
void mythread_sleep(size_t ms);

/**
 * Read a character from user input. If no input is available, the thread should
 * block until input becomes available. The blocked thread should yield the CPU.
 *
 * \returns The read character code
 */
int mythread_readchar();

#endif
