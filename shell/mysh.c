#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_ARGS 128

typedef struct cmd {
  char *command;
  int flag;  // 0 for '&', 1 for ';', and 2 for the end of commands
} cmd;

void run_command(char **commands, bool is_block) {
  if (!strcmp(commands[0], "cd")) {
    chdir(commands[1]);
  } else if (!strcmp(commands[0], "exit")) {
    exit(0);
  } else {
    pid_t child_pid = fork();
    if (child_pid) {  // parent process
      if (is_block) {
        int wstatus;
        waitpid(child_pid, &wstatus, 0);
        if (!WIFEXITED(wstatus)) {
          fprintf(stderr, "Child process did not exit normally.\n");
        } else {
          printf("Child process %d exited with status %d\n", child_pid,
                 WEXITSTATUS(wstatus));
        }
      }
    } else {  // child process
      if (!execvp(commands[0], commands)) {
        fprintf(stderr, "Command not found.\n");
        exit(1);
      }
    }
  }
}

cmd *print_commands(char *line) {
  // Walk through pieces of the string delimited by ampersands or semicolons
  char *pos = line;
  cmd *cmds = (cmd *)malloc(MAX_ARGS * sizeof(cmd));
  int index = 0;
  // Find the next occurrence of a semicolon or ampersand
  while (1) {
    char *end_pos = strpbrk(pos, "&;");

    if (end_pos == NULL) {
      char *temp =
          strdup(pos);  // the last command ends with a '\n' and we need
                        // to get rid of it
      temp[strlen(temp) - 1] = '\0';
      if (strlen(temp) != 0) {  // if it is not an empty command
        cmd c = {.command = temp, .flag = 1};
        cmds[index++] = c;
      }
      cmd last = {.flag = 2};
      cmds[index] = last;
      return cmds;
    } else if (*end_pos == ';') {
      *end_pos = '\0';
      char *temp = strdup(pos);
      cmd c = {.command = temp, .flag = 1};
      cmds[index++] = c;
    } else if (*end_pos == '&') {
      *end_pos = '\0';
      char *temp = strdup(pos);
      cmd c = {.command = temp, .flag = 0};
      cmds[index++] = c;
    } else {
      // This should never happen, but just being thorough
      fprintf(stderr, "Something strange happened!\n");
      exit(1);
    }
    // Move the position pointer to the beginning of the next split
    pos = end_pos + 1;
  }
}

int main(int argc, char **argv) {
  // If there was a command line option passed in, use that file instead of
  // stdin
  if (argc == 2) {
    // Try to open the file
    int new_input = open(argv[1], O_RDONLY);
    if (new_input == -1) {
      fprintf(stderr, "Failed to open input file %s\n", argv[1]);
      exit(1);
    }

    // Now swap this file in and use it as stdin
    if (dup2(new_input, STDIN_FILENO) == -1) {
      fprintf(stderr, "Failed to set new file as input\n");
      exit(2);
    }
  }

  char *line = NULL;     // Pointer that will hold the line we read in
  size_t line_size = 0;  // The number of bytes available in line
  int wstatus;
  pid_t child_pid;
  // Loop forever
  while (1) {
    // Print the shell prompt
    printf("> ");

    // Get a line of stdin, storing the string pointer in line
    if (getline(&line, &line_size, stdin) == -1) {
      if (errno == EINVAL) {
        perror("Unable to read command line");
        exit(2);
      } else {
        // Must have been end of file (ctrl+D)
        printf("\nShutting down...\n");

        // Exit the infinite loop
        break;
      }

    } else {
      cmd *cmds = print_commands(line);
      int i = 0;  // index for cmds
      while (cmds[i].flag != 2) {
        char *cur_cmd = cmds[i].command;
        char *commands[MAX_ARGS + 1];
        int index = 0;
        char *token = strtok(cur_cmd, " ");
        while (token != NULL) {
          commands[index++] = strdup(token);
          token = strtok(NULL, " ");
        }
        commands[index] = NULL;
        if (cmds[i].flag) {  // if the command ends with a semicolon
          run_command(commands, true);
        } else {  // if the command ends with an ampsend
          run_command(commands, false);
        }

        // collecting child processes
        child_pid = waitpid(0, &wstatus, WNOHANG);
        while (child_pid != -1 && child_pid != 0) {
          printf("Child process %d exited with status %d\n", child_pid,
                 WEXITSTATUS(wstatus));
          child_pid = waitpid(0, &wstatus, WNOHANG);
        }

        i++;
      }

      free(cmds);
    }

    /* printf("Received command: %s\n", line); */
  }

  // If we read in at least one line, free this space
  if (line != NULL) {
    free(line);
  }

  return 0;
}
