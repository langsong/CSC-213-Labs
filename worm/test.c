#include <stdio.h>
#include "mythread.h"

void thread_fn() {
  printf("entering thread_fn\n");
  mythread_sleep(1000);
  printf("Done with sleeping\n");
}

int main() {
  mythread_init();
  mythread_t t;
  mythread_create(&t, thread_fn);
  printf("finish creating thread\n");
  mythread_join(t);
  printf("done with everything\n");
  return 0;
}
