#define _GNU_SOURCE
#include <openssl/md5.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define PASSWORD_LENGTH 6

int md5_string_to_bytes(const char *md5_string, uint8_t *bytes);
void print_md5_bytes(const uint8_t *bytes);
void crack_single_password(char *plaintext, char *hash);
void generate_all_possibilities(char *hash);

// Recursion is way too slow to compute
// void generate_all_possibilities(char* guess, int index, char* hash) {
//   if(index == 6) { // Base case
//     printf("string: %s\n", guess);
//     // crack_single_password(guess, hash);
//     return;
//   } else { // Recursive case
//     for(char i = 'z'; i >= 'a'; i--) {
//       // char* new_guess = strdup(guess);
//       guess[index] = i;
//       generate_all_possibilities(guess, index+1, hash);
//       // free(new_guess);
//     }
//   }
// }

void generate_all_possibilities(char *hash) {
  size_t len = pow(26.0, 6);
  char guess[7] = "aaaaaa";
  int cur_digit = 5;
  for (size_t i = 0; i < len; i++) {
    size_t num = i;
    while (num > 0) {
      int remainder = num % 26;
      char digit = remainder + 'a';
      guess[cur_digit--] = digit;
      num = (num - remainder) / 26;
    }
    cur_digit = 5;
    printf("%s\n", guess);
    crack_single_password(guess, hash);
  }
}

void crack_single_password(char *plaintext, char *hash) {
  // This will hold the bytes of our md5 hash input
  uint8_t input_hash[MD5_DIGEST_LENGTH];

  // Convert the string representation of the MD5 hash to a byte array
  md5_string_to_bytes(hash, input_hash);

  uint8_t password_hash[MD5_DIGEST_LENGTH];
  MD5((unsigned char *)plaintext, strlen(plaintext), password_hash);

  // Check if the two hashes are equal
  if (memcmp(input_hash, password_hash, MD5_DIGEST_LENGTH) == 0) {
    printf("%s\n", plaintext);
    // printf("Those two hashes are equal!\n");
    exit(0);
  } else {
    // printf("Those hashes are not equal.\n");
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <md5 sum of %d character password>\n", argv[0],
            PASSWORD_LENGTH);
    exit(1);
  }
  generate_all_possibilities(argv[1]);

  return 0;
}

/**
 * Convert a string representation of an MD5 hash to a sequence
 * of bytes. The input md5_string must be 32 characters long, and
 * the output buffer bytes must have room for MD5_DIGEST_LENGTH
 * bytes.
 *
 * \param md5_string  The md5 string representation
 * \param bytes       The destination buffer for the converted md5 hash
 * \returns           0 on success, -1 otherwise
 */
int md5_string_to_bytes(const char *md5_string, uint8_t *bytes) {
  // Check for a valid MD5 string
  if (strlen(md5_string) != 2 * MD5_DIGEST_LENGTH)
    return -1;

  // Start our "cursor" at the start of the string
  const char *pos = md5_string;

  // Loop until we've read enough bytes
  for (size_t i = 0; i < MD5_DIGEST_LENGTH; i++) {
    // Read one byte (two characters)
    int rc = sscanf(pos, "%2hhx", &bytes[i]);
    if (rc != 1)
      return -1;

    // Move the "cursor" to the next hexadecimal byte
    pos += 2;
  }

  return 0;
}

/**
 * Print a byte array that holds an MD5 hash to standard output.
 *
 * \param bytes   An array of bytes from an MD5 hash function
 */
void print_md5_bytes(const uint8_t *bytes) {
  for (size_t i = 0; i < MD5_DIGEST_LENGTH; i++) {
    printf("%02hhx", bytes[i]);
  }
}
