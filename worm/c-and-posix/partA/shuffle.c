#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Here are some handy C preprocessor definitions for suits of cards
// Source: http://stackoverflow.com/questions/27133508/how-to-print-spades-hearts-diamonds-etc-in-c-and-linux
#if defined(_WIN32) || defined(__MSDOS__)
   #define SPADE   "\x06"
   #define CLUB    "\x05"
   #define HEART   "\x03"
   #define DIAMOND "\x04"
#else
   #define SPADE   "\xE2\x99\xA0"
   #define CLUB    "\xE2\x99\xA3"
   #define HEART   "\xE2\x99\xA5"
   #define DIAMOND "\xE2\x99\xA6"
#endif

//Fisher-Yates Shuffle on int array of size 52
void shuffle(int* cards) {
  for(int i = 51; i >= 0; i--) {
    int j = rand() % (i+1);
    int temp = cards[i];
    cards[i] = cards[j];
    cards[j] = temp;
  }
}

// mapping function takes int array of size 52 and
// prints out their corresponding cards
// ex. 5 = 6 SPADES
void transform(int* cards) {
  for(int i = 0; i < 52; i++) {
    // map to card suit
    int suits = cards[i] / 13;
    // map to card number
    int number = cards[i] % 13 + 1;
    char altNum[2];
    sprintf(altNum, "%d", number);
      // switch statement to convert 1, 11, 12, 13
      // to A, J, Q, K respectively
    switch(number) {
    case 1:
      altNum[0] = 'A';
      break;
    case 11:
      altNum[0] = 'J';
      altNum[1] = '\0';
      break;
    case 12:
      altNum[0] = 'Q';
      altNum[1] = '\0';
      break;
    case 13:
      altNum[0] = 'K';
      altNum[1] = '\0';
      break;
    }
    // switch statement to section deck into suits
    switch(suits) {
    case 0:
      printf("%s"SPADE"\n", altNum);
      break;
    case 1:
      printf("%s"CLUB"\n", altNum);
      break;
    case 2:
      printf("%s"HEART"\n", altNum);
      break;
    case 3:
      printf("%s"DIAMOND"\n", altNum);
      break;
    }
  }
}


int main(int argc, char** argv) {
  printf("To Do: shuffle a deck of 52 cards.\n");
  srand (time(NULL));

  //initialize deck of cards
  int cards[52];
  for(int i = 0; i < 52; i++) {
    cards[i] = i;
  }

  shuffle(cards);
  transform(cards);
  // These demo printouts do something horrible with the C preprocessor.
  // Adjacent strings in C are just combined into one.
  // printf("King of spades: K" SPADE "\n");
  // printf("Two of clubs: 2" CLUB "\n");
  // printf("Ace of hearts: A" HEART "\n");
  // printf("Jack of diamonds: J" DIAMOND "\n");
  return 0;
}
