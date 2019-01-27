#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED

#include "mythread.h"

#include <assert.h>
#include <curses.h>
#include <pthread.h>
#include <ucontext.h>
#include <stdbool.h>

#include "util.h"

// Status
#define is_running 1
#define blocked_on_input 2
#define blocked_on_sleep 3
#define blocked_on_join 4
#define exited 5
#define ready 6

// This is an upper limit on the number of threads we can create.
#define MAX_THREADS 128

// This is the size of each thread's stack memory
#define STACK_SIZE 65536

// This struct will hold the all the necessary information for each thread
typedef struct mythread_info {
  // This field stores all the state required to switch back to this thread
  ucontext_t context;

  // This field stores another context. This one is only used when the thread
  // is exiting.
  ucontext_t exit_context;
  int status;
  size_t wake_time;
  mythread_t join_thread;
  int input;
  // TODO: Add fields here so you can:
  //   a. Keep track of this thread's state.
  //   b. If the thread is sleeping, when should it wake up?
  //   c. If the thread is waiting to join, which thread is it waiting for?
  //   d. Was the thread blocked waiting for user input? Once you successfully
  //      read input, you will need to save it here so it can be returned.
} mythread_info_t;

int current_thread = 0; //< The handle of the currently-executing thread
int num_threads = 1;    //< The number of threads created so far
mythread_info_t threads[MAX_THREADS]; //< Information for every thread

void schedule(mythread_t t) {
  mythread_info_t cur_thread = threads[t];
  mythread_t index = (t + 1) % num_threads;
  while(true) {
    if(threads[index].status == ready) {
      threads[index].status = is_running;
      current_thread = index;
      printf("here\n");
      swapcontext(&cur_thread.context, &threads[index].context);
      return;
    } else if(threads[index].status == blocked_on_sleep){
      if(time_ms() >= threads[index].wake_time) {
        threads[index].status = is_running;
        current_thread = index;
        swapcontext(&(cur_thread.context), &(threads[index].context));
        return;
      }
    } else if(threads[index].status == blocked_on_join) {
      if(t == threads[index].join_thread) {
        threads[index].status = is_running;
        current_thread = index;
        swapcontext(&(cur_thread.context), &(threads[index].context));
        return;
      }
    } else if(threads[index].status == blocked_on_input) {
      int res = getch();
      if(res != ERR) {
        threads[index].status = is_running;
        current_thread = index;
        threads[index].input = res;
        swapcontext(&(cur_thread.context), &(threads[index].context));
        return;
      }
    }

    index = (index + 1) % num_threads;
    sleep_ms(1);
  }
}

/**
 * Initialize the mythread library. Programs should call this before calling
 * any other mythread functions.
 */
void mythread_init() {
  current_thread = 0;
  num_threads = 1;
  threads[0].status = is_running;
  getcontext(&threads[0].context);
}

/**
 * This function will execute when a thread's function returns. This allows you
 * to update scheduler states and start another thread. This function is run
 * because of how the contexts are set up in the mythread_create function.
 */
void end_thread() {
  // TODO: Handle the end of a running thread here
  threads[current_thread].status = exited;
  schedule(current_thread);
}

/**
 * Create a new thread.
 *
 * \param handle  The handle for this thread will be written to this location.
 * \param fn      The new thread will run this function.
 */
void mythread_create(mythread_t* handle, mythread_fn_t fn) {
  // Claim an index for the new thread
  int index = num_threads;
  num_threads++;

  // Set the thread handle to this index, since mythread_t is just an int
  *handle = index;

  // We need to create a context for the new thread to use. It's easiest to
  // start with the current thread's context and then modify it as needed
  getcontext(&threads[index].context);

  // Allocate a stack for the new thread and add it to the context
  threads[index].context.uc_stack.ss_sp = malloc(STACK_SIZE);
  threads[index].context.uc_stack.ss_size = STACK_SIZE;

  // Now we're going to set up *another* context. This one will execute when our new thread exits.
  getcontext(&threads[index].exit_context);
  threads[index].exit_context.uc_stack.ss_sp = malloc(STACK_SIZE);
  threads[index].exit_context.uc_stack.ss_size = STACK_SIZE;
  makecontext(&threads[index].exit_context, end_thread, 0);
  // To get our new thread to switch to the exiting context when the thread
  // function returns, we set it in the uc_link field of the context
  threads[index].context.uc_link = &threads[index].exit_context;


  // Set up the context to execute the thread function
  makecontext(&threads[index].context, fn, 0);

  threads[index].status = ready;
}

/**
 * Join with a thread.
 *
 * \param handle  This is the handle produced by mythread_create.
 */
void mythread_join(mythread_t handle) {
  // TODO: Block this thread until the specified thread has exited.
  threads[current_thread].status = blocked_on_join;
  threads[current_thread].join_thread = handle;
  schedule(current_thread);
}

/**
 * The currently-executing thread should sleep and yield the CPU.
 *
 * \param ms  The number of milliseconds the thread should sleep.
 */
void mythread_sleep(size_t ms) {
  // TODO: Block this thread until the requested time has elapsed.
  // Hint: Record the time the thread should wake up instead of the time left for it to sleep. The bookkeeping is easier this way.
  threads[current_thread].status = blocked_on_sleep;
  threads[current_thread].wake_time = time_ms() + ms;
  schedule(current_thread);
}

/**
 * Read a character from user input. If no input is available, the thread should
 * block until input becomes available. The blocked thread should yield the CPU.
 *
 * \returns The read character code
 */
int mythread_readchar() {
  // TODO: Block this thread until there is input available for the thread.
  // Call the function getch(). If it returns ERR, no input was available.
  // Otherwise it returns the character code that was read.
  threads[current_thread].status = blocked_on_input;
  schedule(current_thread);
  return threads[current_thread].input;
}
