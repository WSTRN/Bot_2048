#include "teacher_2048.h"
#include <catch2/catch.hpp>

TEST_CASE("print board") {
    board_t board = 0x0002000400080001ULL;
    print_board(board);
    REQUIRE(true);
}
