#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

// How much does one pizza cost?
#define PIZZA_COST 16

// Prompt for an input line
char* prompt(char* message) {
  printf("%s", message);
  char* line = NULL;
  size_t linecap = 0;
  if(getline(&line, &linecap, stdin) == NULL) {
    perror("getline failed");
    exit(2);
  }
  
  // Remove the newline
  line[strlen(line)-1] = '\0';
  
  return line;
}

int main(int argc, char** argv) {
  printf("Welcome to the pizza shop.\n");
  printf("Delicious pizzas cost $%u each. There are no free pizzas!\n", PIZZA_COST);

  // Prompt the user for an input number
  char* line = prompt("How many pizzas would you like to buy?\n");

  // Parse the number
  uint8_t pizza_count = atoi(line);
  
  // Clean up after prompt
  free(line);

  // Handle the order
  if(pizza_count < 1 || pizza_count >= 100) {
    printf("That's a ridiculous number of pizzas. Go away!\n");
  } else {
    printf("Thank you for your order!\n");
    
    // Make the pizzas and charge for each one
    uint8_t total_cost = 0;
    for(uint8_t i=1; i<=pizza_count; i++) {
      printf("  Making pizza %u.\n", i);
      usleep(250000); // These pizzas cook really fast
      total_cost += PIZZA_COST;
    }

    // Report the total
    printf("The total for your order is $%u.\n", total_cost);
  }

  return 0;
}

