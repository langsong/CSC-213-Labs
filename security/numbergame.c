#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char** argv) {
  // Seed the random number generator
  srand(time(NULL));

  // Generate a random number from 1 to 10
  int secret = rand() % 9 + 1;

  // Display a prompt
  printf("I'm thinking of a random number from 1 to 10. Try to guess it.\n");

  // Read in the number. Ten characters should be enough for any 32-bit number plus a null terminator
  char buffer[10];
  if(gets(buffer) == NULL) {
    perror("gets failed");
    exit(2);
  }

  // Check the guess
  if(atoi(buffer) == secret) {
    printf("You guessed it.\n");
  } else {
    printf("Wrong. You lose!\n");
  }

  return 0;
}

