#include <gtest/gtest.h>
#include <future>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>

#include "stack.h"

using namespace std;

// A simple test of basic stack functionality
TEST(StackTest, BasicStackOps) {
  // Create a stack
  my_stack_t s;
  stack_init(&s);
  
  // Push some values onto the stack
  stack_push(&s, 1);
  stack_push(&s, 2);
  stack_push(&s, 3);
  
  // Make sure the elements come off the stack in the right order
  ASSERT_EQ(3, stack_pop(&s));
  ASSERT_EQ(2, stack_pop(&s));
  ASSERT_EQ(1, stack_pop(&s));
  
  // Clean up
  stack_destroy(&s);
}

// Another test case
TEST(StackTest, EmptyStack) {
  // Create a stack
  my_stack_t s;
  stack_init(&s);
  
  // The stack should be empty
  ASSERT_TRUE(stack_empty(&s));
  
  // Popping an empty stack should return -1
  ASSERT_EQ(-1, stack_pop(&s));
  
  // Put something on the stack
  stack_push(&s, 0);
  
  // The stack should not be empty
  ASSERT_FALSE(stack_empty(&s));
  
  // Pop the element off the stack.
  // We're just testing empty stack behavior, so there's no need to check the resulting value
  stack_pop(&s);
  
  // The stack should be empty now
  ASSERT_TRUE(stack_empty(&s));
  
  // Clean up
  stack_destroy(&s);
}

int thread_push_inv1(my_stack_t* s, int num, size_t num_times) {
  for(size_t i = 0; i < num_times; i++) {
    stack_push(s, num);
  }
  return 1;
}

int thread_push(my_stack_t* s, int start, size_t end) {
  for(size_t i = start; i < end; i++) {
    stack_push(s, i);
  }
  return 1;
}

int thread_pop(my_stack_t* s, size_t num_times) {
  int result;
  for(size_t i = 0; i < num_times; i++) {
    result += stack_pop(s);
  }
  return result;
}

int thread_pop_inv3(my_stack_t* s, size_t num_times, vector<int> *v) {
  for(size_t i = 0; i < num_times; i++) {
    v->push_back(stack_pop(s));
  }
  return 1;
}

TEST(StackTest, Invariant1) {
  // Create a stack
  my_stack_t s;
  stack_init(&s);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_push_inv1, &s, 17, 1000);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_pop, &s, 900);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }
  
  for(size_t i = 0; i < 400; i++) {
    ASSERT_FALSE(stack_empty(&s));
    stack_pop(&s);
  }
  ASSERT_TRUE(stack_empty(&s));
  
  // Clean up
  stack_destroy(&s);
}

TEST(StackTest, Invariant2) {
  // Create a stack
  my_stack_t s;
  stack_init(&s);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_push, &s, i * 100, (i+1) * 100);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  future<int> results[4];
  for(size_t i = 0; i < num_threads; i++) {
    results[i] = async(thread_pop, &s, 100);
  }
  int final_sum = 0;
  for(size_t i = 0; i < 4; i++) {
    int res = results[i].get();
    final_sum += res;
  }
  ASSERT_TRUE(final_sum == 79800);
  ASSERT_TRUE(stack_empty(&s));
  
  // Clean up
  stack_destroy(&s);
}

TEST(StackTest, Invariant3) {
  // Create a stack
  my_stack_t s;
  stack_init(&s);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < 1000; i++) {
    stack_push(&s, i);
  }

  vector<int> vs[4];
  for(size_t i = 0; i < num_threads; i++) {
    auto result = async(thread_pop_inv3, &s, 250, &vs[i]);
  }
  for(size_t i = 0; i < num_threads; i++) {
    ASSERT_TRUE(is_sorted(vs[i].begin(), vs[i].end(), greater<int>()));
  }

  ASSERT_TRUE(stack_empty(&s));
  
  // Clean up
  stack_destroy(&s);
}
