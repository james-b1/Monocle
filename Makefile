CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -fPIC
LDFLAGS  := -shared

SRC_DIR  := cpp/src
INC_DIR  := cpp/include
BUILD    := build

LIB_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS := $(LIB_SRCS:$(SRC_DIR)/%.cpp=$(BUILD)/%.o)
LIB      := $(BUILD)/libmonocle.dylib

BENCH_SRC := cpp/bench/bench.cpp
BENCH_BIN := $(BUILD)/bench

.PHONY: all bench clean
all: $(LIB)
bench: $(BENCH_BIN)

$(LIB): $(LIB_OBJS) | $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c -o $@ $<

$(BENCH_BIN): $(BENCH_SRC) $(BUILD)/search.o | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -I$(SRC_DIR) -o $@ $^

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
