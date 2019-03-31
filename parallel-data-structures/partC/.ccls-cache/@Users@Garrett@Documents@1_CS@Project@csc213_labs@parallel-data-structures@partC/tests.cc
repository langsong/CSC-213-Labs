#include <gtest/gtest.h>

#include <stdlib.h>
#include <time.h>
#include <future>
#include <thread>
#include <vector>
#include "dict.h"

using namespace std;

// Basic functionality for the dictionary
TEST(DictionaryTest, BasicDictionaryOps) {
  my_dict_t d;
  dict_init(&d);

  // Make sure the dictionary does not contain keys A, B, and C
  ASSERT_FALSE(dict_contains(&d, "A"));
  ASSERT_FALSE(dict_contains(&d, "B"));
  ASSERT_FALSE(dict_contains(&d, "C"));

  // Add some values
  dict_set(&d, "A", 1);
  dict_set(&d, "B", 2);
  dict_set(&d, "C", 3);

  // Make sure these values are contained in the dictionary
  ASSERT_TRUE(dict_contains(&d, "A"));
  ASSERT_TRUE(dict_contains(&d, "B"));
  ASSERT_TRUE(dict_contains(&d, "C"));

  // Make sure these values are in the dictionary
  ASSERT_EQ(1, dict_get(&d, "A"));
  ASSERT_EQ(2, dict_get(&d, "B"));
  ASSERT_EQ(3, dict_get(&d, "C"));

  // Set some new values
  dict_set(&d, "A", 10);
  dict_set(&d, "B", 20);
  dict_set(&d, "C", 30);

  // Make sure these values are contained in the dictionary
  ASSERT_TRUE(dict_contains(&d, "A"));
  ASSERT_TRUE(dict_contains(&d, "B"));
  ASSERT_TRUE(dict_contains(&d, "C"));

  // Make sure the new values are in the dictionary
  ASSERT_EQ(10, dict_get(&d, "A"));
  ASSERT_EQ(20, dict_get(&d, "B"));
  ASSERT_EQ(30, dict_get(&d, "C"));

  // Remove the values
  dict_remove(&d, "A");
  dict_remove(&d, "B");
  dict_remove(&d, "C");

  // Make sure these values are not contained in the dictionary
  ASSERT_FALSE(dict_contains(&d, "A"));
  ASSERT_FALSE(dict_contains(&d, "B"));
  ASSERT_FALSE(dict_contains(&d, "C"));

  // Make sure we get -1 for each value
  ASSERT_EQ(-1, dict_get(&d, "A"));
  ASSERT_EQ(-1, dict_get(&d, "B"));
  ASSERT_EQ(-1, dict_get(&d, "C"));

  // Clean up
  dict_destroy(&d);
}

int thread_insert(my_dict_t* s, vector<const char*> v, int start, int end) {
  for (size_t i = start; i < end; i++) {
    dict_set(s, v[i], i);
  }
  return 1;
}

int thread_insert_inv3(my_dict_t* s, vector<const char*> v, int start,
                       int end) {
  for (size_t i = start; i < end; i++) {
    dict_set(s, v[i], i + 1);
  }
  return 1;
}

int thread_get(my_dict_t* s, vector<const char*> v, int start, int end) {
  int result;
  for (size_t i = start; i < end; i++) {
    int temp = dict_get(s, v[i]);
    result += temp;
  }
  return result;
}

TEST(DictTest, Invariant1) {
  srand(time(0));
  // Create a dict
  my_dict_t s;
  dict_init(&s);
  const int num_threads = 4;
  const int num_entry = 10000;
  thread t[num_threads];

  vector<const char*> v;

  for (int i = 0; i < num_threads * num_entry; i++) {
    char* res = (char*)malloc(21 * sizeof(char));
    for (int i = 0; i < 20; i++) {
      res[i] = 97 + rand() % 26;
    }
    res[20] = '\0';
    v.push_back(res);
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_insert, &s, v, num_entry * i, num_entry * (i + 1));
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  future<int> results[4];
  for (size_t i = 0; i < num_threads; i++) {
    results[i] = async(thread_get, &s, v, num_entry * i, num_entry * (i + 1));
  }
  int final_sum = 0;
  for (size_t i = 0; i < num_threads; i++) {
    int res = results[i].get();
    final_sum += res;
  }

  ASSERT_EQ(799980000, final_sum);
  for (int i = 0; i < num_entry * num_threads; i++) {
    free((void*)v[i]);
  }

  // Clean up
  dict_destroy(&s);
}

TEST(DictTest, Invariant2) {
  srand(time(0));
  // Create a dict
  my_dict_t s;
  dict_init(&s);
  const int num_threads = 4;
  const int num_entry = 10000;
  thread t[num_threads];

  vector<const char*> v;
  vector<const char*> different_v;

  for (int i = 0; i < num_threads * num_entry; i++) {
    char* res = (char*)malloc(21 * sizeof(char));
    for (int i = 0; i < 20; i++) {
      res[i] = 97 + rand() % 26;
    }
    res[20] = '\0';
    v.push_back(res);
  }

  for (int i = 0; i < num_threads * num_entry; i++) {
    char* res = (char*)malloc(21 * sizeof(char));
    for (int i = 0; i < 20; i++) {
      res[i] = 97 + rand() % 26;
    }
    res[20] = '\0';
    different_v.push_back(res);
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_insert, &s, v, num_entry * i, num_entry * (i + 1));
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  future<int> results[4];
  for (size_t i = 0; i < num_threads; i++) {
    results[i] =
        async(thread_get, &s, different_v, num_entry * i, num_entry * (i + 1));
  }
  int final_sum = 0;
  for (size_t i = 0; i < num_threads; i++) {
    int res = results[i].get();
    final_sum += res;
  }

  ASSERT_EQ(num_entry * num_threads * -1, final_sum);
  for (int i = 0; i < num_entry * num_threads; i++) {
    free((void*)v[i]);
  }

  // Clean up
  dict_destroy(&s);
}

TEST(DictTest, Invariant3) {
  srand(time(0));
  // Create a dict
  my_dict_t s;
  dict_init(&s);
  const int num_threads = 4;
  const int num_entry = 10000;
  thread t[num_threads];

  vector<const char*> v;

  for (int i = 0; i < num_threads * num_entry; i++) {
    char* res = (char*)malloc(21 * sizeof(char));
    for (int i = 0; i < 20; i++) {
      res[i] = 97 + rand() % 26;
    }
    res[20] = '\0';
    v.push_back(res);
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i] = thread(thread_insert, &s, v, num_entry * i, num_entry * (i + 1));
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i] =
        thread(thread_insert_inv3, &s, v, num_entry * i, num_entry * (i + 1));
  }

  for (size_t i = 0; i < num_threads; i++) {
    t[i].join();
  }

  future<int> results[4];
  for (size_t i = 0; i < num_threads; i++) {
    results[i] = async(thread_get, &s, v, num_entry * i, num_entry * (i + 1));
  }
  int final_sum = 0;
  for (size_t i = 0; i < num_threads; i++) {
    int res = results[i].get();
    final_sum += res;
  }

  ASSERT_EQ(799980000 + num_entry * num_threads, final_sum);
  for (int i = 0; i < num_entry * num_threads; i++) {
    free((void*)v[i]);
  }

  // Clean up
  dict_destroy(&s);
}
