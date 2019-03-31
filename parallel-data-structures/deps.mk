# Targets to fetch and build external dependencies

# Get and build libelfin
$(ROOT)/deps/gtest:
	@echo $(LOG_PREFIX) Downloading gtest $(LOG_SUFFIX)
	@mkdir -p $(ROOT)/deps
	@cd $(ROOT)/deps; \
		wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz; \
		tar xzf release-1.8.1.tar.gz; \
		rm release-1.8.1.tar.gz; \
		mv googletest-release-1.8.1 gtest

# Update build settings to include the gtest framework
ifneq (,$(findstring gtest,$(PREREQS)))
CXXFLAGS += -isystem $(ROOT)/deps/gtest/googletest/include/
LDFLAGS += -isystem $(ROOT)/deps/gtest/googletest/ \
					 -isystem $(ROOT)/deps/gtest/googletest/include/ \
					 $(ROOT)/deps/gtest/googletest/src/gtest-all.cc \
					 $(ROOT)/deps/gtest/googletest/src/gtest_main.cc
endif
