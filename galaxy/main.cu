#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <SDL.h>

#include "gui.h"

// Time step size
#define DT 0.075

// Gravitational constant
#define G 100

#define BLOCK_SIZE 32

// This struct holds data for a single star
typedef struct star {
  float x_position;
  float y_position;
  float x_velocity;
  float y_velocity;
  float mass;
} star_t;

// Generate a random float in a given range
float drand(float min, float max) {
  return ((float)rand() / RAND_MAX) * (max - min) + min;
}

// Compute the radius of a star based on its mass
__device__ __host__ float star_radius(float mass) {
  return sqrt(mass);
}

__global__ void updatePosition(star_t* stars, int num_stars) {
  int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  if(index < num_stars) {
    stars[index].x_position += stars[index].x_velocity * DT;
    stars[index].y_position += stars[index].y_velocity * DT;
    // Loop over all other stars to compute forces
    for(int j=0; j<num_stars; j++) {
      // Don't compute the force of a star on itself
      if(index == j) continue;

      // Compute the distance between the two stars in each dimension
      float x_diff = stars[index].x_position - stars[j].x_position;
      float y_diff = stars[index].y_position - stars[j].y_position;

      // Compute the magnitude of the distance vector
      float dist = sqrt(x_diff * x_diff + y_diff * y_diff);

      // Normalize the distance vector components
      x_diff /= dist;
      y_diff /= dist;

      // Keep a minimum distance, otherwise we get 
      float combined_radius = star_radius(stars[index].mass) + star_radius(stars[j].mass);
      if(dist < combined_radius) {
        dist = combined_radius;
      }

      // Compute the x and y accelerations
      float x_acceleration = -x_diff * G * stars[j].mass / (dist * dist);
      float y_acceleration = -y_diff * G * stars[j].mass / (dist * dist);

      // Update the star velocity
      stars[index].x_velocity += x_acceleration * DT;
      stars[index].y_velocity += y_acceleration * DT;

      // Handle edge collisiosn
      if(stars[index].x_position < 0 && stars[index].x_velocity < 0)
        stars[index].x_velocity *= -0.5;
      if(stars[index].x_position >= SCREEN_WIDTH && stars[index].x_velocity
          > 0) stars[index].x_velocity *= -0.5;
      if(stars[index].y_position < 0 && stars[index].y_velocity < 0) stars[index].y_velocity *= -0.5;
      if(stars[index].y_position >= SCREEN_HEIGHT && stars[index].y_velocity
          > 0) stars[index].y_velocity *= -0.5;
    }
  }
}



int main(int argc, char** argv) {
  // Initialize the graphical interface
  gui_init();

  // Run as long as this is true
  bool running = true;

  // Is the mouse currently clicked?
  bool clicked = false;

  // This will hold our array of stars
  star_t* stars = NULL;
  int num_stars = 0;
  star_t* gpu_stars = NULL;
  int prev_num_stars = -1;

  // Start main loop
  while(running) {
    // Check for events
    SDL_Event event;
    while(SDL_PollEvent(&event) == 1) {
      // If the event is a quit event, then leave the loop
      if(event.type == SDL_QUIT) running = false;
    }

    // Get the current mouse state
    int mouse_x, mouse_y;
    uint32_t mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);

    // Is the mouse pressed?
    if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      // Is this the beginning of a mouse click?
      if(!clicked) {
        clicked = true;
        stars = (star_t*)realloc(stars, (num_stars + 1) * sizeof(star_t));
        stars[num_stars].x_position = mouse_x;
        stars[num_stars].y_position = mouse_y;
        stars[num_stars].x_velocity = 0;
        stars[num_stars].y_velocity = 0;
        // Generate a random mass skewed toward small sizes
        stars[num_stars].mass = drand(0, 1) * drand(0, 1) * 50;
        num_stars++;
      }
    } else {
      // The mouse click is finished
      clicked = false;
    }
    if(prev_num_stars != num_stars) {
      if(prev_num_stars != -1) {
        cudaFree(gpu_stars);
      }
      if(cudaMalloc(&gpu_stars, sizeof(star_t) * num_stars) != cudaSuccess) {  
        fprintf(stderr, "Failed to allocate X array on GPU\n");
        exit(2);
      }
      prev_num_stars = num_stars;
    }

    // Draw stars
    for(int i=0; i<num_stars; i++) {
      color_t color = {255, 255, 255, 255};
      gui_draw_circle(stars[i].x_position, stars[i].y_position, star_radius(stars[i].mass), color);
    }

    // Calculate the number of blocks we need
    size_t blocks = (num_stars + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Copy from CPU to GPU
    if(cudaMemcpy(gpu_stars, stars, sizeof(star_t) * num_stars, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Failed to copy stars to the GPU\n");
    }
    // Update the position in GPU
    updatePosition<<<blocks, BLOCK_SIZE>>>(gpu_stars, num_stars);
    // Wait untill all threads join
    cudaDeviceSynchronize();

    // Copy from GPU to CPU
    if(cudaMemcpy(stars,gpu_stars, sizeof(star_t) * num_stars,
          cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "Failed to copy gpu_stars from the GPU\n");
    }

    // Display the rendered image
    gui_update_display();

    // Fade out the rendered image to leave movement trails
    gui_fade(0.85);
  }

  // Free the stars array
  free(stars);

  // Clean up the graphical interface
  gui_shutdown();

  return 0;
}
