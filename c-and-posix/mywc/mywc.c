#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTES_READ 4  // number of bytes read in each time

int main(int argc, char **argv) {
  char countThis[BYTES_READ];
  long counter = 0;                // global counter for the final word count
  char *delimeter = " \t\n";       // three delimeters
  bool isLastDigitLetter = false;  // a boolean check to see if
                                   // the last digit is a letter
  while (fgets(countThis, BYTES_READ, stdin)) {
    char *token;

    char *dup = strdup(countThis);
    // use strtok() to parse the current string
    token = strtok(dup, delimeter);

    // use a temp counter for each string
    long tempCounter = 0;
    while (token != NULL) {
      tempCounter++;
      token = strtok(NULL, delimeter);
    }

    free(dup);

    // if the previous string ends with a character
    // and this string starts with a character,
    // we need to subtract one from the counter
    if (isLastDigitLetter && countThis[0] != '\t' && countThis[0] != ' ' &&
        countThis[0] != '\n') {
      counter--;
    }

    // change this boolean back to default: false
    isLastDigitLetter = false;

    // if we found at least one token in this string
    // and the string ends with a character,
    // we set the boolean to be true
    if (countThis[BYTES_READ - 2] != '\t' && countThis[BYTES_READ - 2] != ' ' &&
        countThis[BYTES_READ - 2] != '\n') {
      isLastDigitLetter = true;
    }

    // add temp counter back to the global counter
    counter += tempCounter;
    memset(countThis, 0, BYTES_READ);
  }

  printf("%ld\n", counter);
  return 0;
}
