#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <time.h>

void sega_handler(int signal, siginfo_t* info, void* ctx);

__attribute__((constructor)) void init() {
  //printf("This code runs at program startup\n");
  srand(time(NULL));
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sa.sa_sigaction = sega_handler;
  sa.sa_flags = SA_SIGINFO;

  if(sigaction(SIGSEGV, &sa, NULL)) {
    perror("sigaction failed");
    exit(2);
  }
}

void sega_handler(int signal, siginfo_t* info, void* ctx) {
  int num = rand() % 8;
  switch(num){
  case 0:
    printf("SegFault: You got this!\n");
    break;
  case 1:
    printf("SegFault: You can do this!\n");
    break;
  case 2:
    printf("SegFault: Try again!\n");
    break;
  case 3:
    printf("SegFault: Sorry! But you are close!\n");
    break;
  case 4:
    printf("SegFault: Check your memory!\n");
    break;
  case 5:
    printf("SegFault: Dangling pointer? Maybe?\n");
    break;
  case 6:
    printf("SegFault: Buffer overflow? Maybe?\n");
    break;
  case 7:
    printf("SegFault: Really?:(\n");
    break;
  default:
    break;
  }
  exit(1);    
}


