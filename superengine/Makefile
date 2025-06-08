CXX ?= g++
BUILD_DIR ?= make-build
CXXFLAGS ?= -std=c++20 -O3 -march=native -flto -Iengine
LDFLAGS ?= -pthread

SRC = engine/bitboard.cpp engine/movegen.cpp engine/nnue_eval.cpp engine/position.cpp engine/search.cpp engine/uci.cpp main.cpp
OBJ = $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))

all: $(BUILD_DIR)/superengine
	$(CXX) $(CXXFLAGS) $(OBJ) $(LDFLAGS) -o $(BUILD_DIR)/superengine

$(BUILD_DIR)/%.o: engine/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
