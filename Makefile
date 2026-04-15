CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -fPIC
LDFLAGS  := -shared

SRC_DIR  := cpp/src
INC_DIR  := cpp/include
BUILD    := build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD)/%.o)
LIB  := $(BUILD)/libmonocle.dylib

.PHONY: all clean
all: $(LIB)

$(LIB): $(OBJS) | $(BUILD)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
