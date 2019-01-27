#include <gtest/gtest.h>

#include "queue.h"
#include <vector>
#include <thread>
#include <future>
#include <algorithm>

using namespace std;

// Basic queue functionality
TEST(QueueTest, BasicQueueOps) {
  my_queue_t q;
  queue_init(&q);
  
  // Make sure the queue is empty
  ASSERT_TRUE(queue_empty(&q));
  
  // Make sure taking from the queue returns -1
  ASSERT_EQ(-1, queue_take(&q));
  
  // Add some items to the queue
  queue_put(&q, 1);
  queue_put(&q, 2);
  queue_put(&q, 3);
  
  // Make sure the queue is not empty
  ASSERT_FALSE(queue_empty(&q));
  
  // Take the values from the queue and check them
  ASSERT_EQ(1, queue_take(&q));
  ASSERT_EQ(2, queue_take(&q));
  ASSERT_EQ(3, queue_take(&q));
  
  // Make sure the queue is empty
  ASSERT_TRUE(queue_empty(&q));
  
  // Clean up
  queue_destroy(&q);
}

int thread_put_inv1(my_queue_t* q, int num, size_t num_times) {
  for(size_t i = 0; i < num_times; i++) {
    queue_put(q, num);
  }
  return 1;
}

int thread_put_inv2(my_queue_t* q, int start, size_t end) {
  for(size_t i = start; i < end; i++) {
    queue_put(q, i);
  }
  return 1;
}

int thread_take(my_queue_t* q, size_t num_times) {
  int result;
  for(size_t i = 0; i < num_times; i++) {
    result += queue_take(q);
  }
  return result;
}

int thread_take_inv3(my_queue_t* q, size_t num_times, vector<int> *v) {
  for(size_t i = 0; i < num_times; i++) {
    v->push_back(queue_take(q));
  }
  return 1;
}

TEST(QueueTest, Invariant1) {
  // Create a stack
  my_queue_t q;
  queue_init(&q);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_put_inv1, &q, 17, 1000);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_take, &q, 900);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }
  
  for(size_t i = 0; i < 400; i++) {
    ASSERT_FALSE(queue_empty(&q));
    queue_take(&q);
  }
  ASSERT_TRUE(queue_empty(&q));
  
  // Clean up
  queue_destroy(&q);
}

TEST(QueueTest, Invariant2) {
  // Create a stack
  my_queue_t q;
  queue_init(&q);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_put_inv2, &q, i * 1000, (i+1) * 1000);
  }
  
  for(size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  future<int> results[4];
  for(size_t i = 0; i < num_threads; i++) {
    results[i] = async(thread_take, &q, 1000);
  }
  int final_sum = 0;
  for(size_t i = 0; i < 4; i++) {
    int res = results[i].get();
    final_sum += res;
  }
  ASSERT_TRUE(final_sum == 7998000);
  ASSERT_TRUE(queue_empty(&q));
  
  // Clean up
  queue_destroy(&q);
}

TEST(QueueTest, Invariant3) {
  // Create a stack
  my_queue_t q;
  queue_init(&q);
  const int num_threads = 4;
  thread t[num_threads];
  
  for(size_t i = 0; i < 1000; i++) {
    queue_put(&q, i);
  }

  vector<int> vs[4];
  for(size_t i = 0; i < num_threads; i++) {
    auto result = async(thread_take_inv3, &q, 250, &vs[i]);
  }
  for(size_t i = 0; i < num_threads; i++) {
    ASSERT_TRUE(is_sorted(vs[i].begin(), vs[i].end()));
  }

  ASSERT_TRUE(queue_empty(&q));
  
  // Clean up
  queue_destroy(&q);
}

