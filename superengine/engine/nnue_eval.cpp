#include "nnue_eval.h"
#include "position.h"
#include <fstream>
#include <array>
#include <cstring>

nnue::Network nnue::net;

static int16_t clamp_relu(int32_t x) { return x > 0 ? (x < 32767 ? x : 32767) : 0; }

static int feature_index(Color c, Piece p, int sq){
    int mirror = c==WHITE ? sq : (sq ^ 56);
    return c*6*64 + p*64 + mirror;
}

static void extract_features(const Position& pos, std::array<int16_t, nnue::INPUTS>& out){
    out.fill(0);
    for(int c=0;c<2;++c){
        for(int p=0;p<PIECE_NB;++p){
            Bitboard bb = pos.piece_bb[c][p];
            while(bb){
                int sq = bb::pop_lsb(bb);
                out[feature_index((Color)c,(Piece)p,sq)] = 1;
            }
        }
    }
}

int nnue::eval(const Position& pos) {
    std::array<int16_t, INPUTS> feats;
    extract_features(pos, feats);
    std::array<int32_t, HIDDEN1> h1{};
    for(int i=0;i<INPUTS;++i){
        if(!feats[i]) continue;
        for(int j=0;j<HIDDEN1;++j){
            h1[j] += net.w1[i*HIDDEN1 + j];
        }
    }
    for(int j=0;j<HIDDEN1;++j) h1[j] = clamp_relu(h1[j]);
    std::array<int32_t, HIDDEN2> h2{};
    for(int i=0;i<HIDDEN1;++i){
        for(int j=0;j<HIDDEN2;++j){
            h2[j] += h1[i] * net.w2[i*HIDDEN2 + j];
        }
    }
    for(int j=0;j<HIDDEN2;++j) h2[j] = clamp_relu(h2[j] + net.bias2[j]);
    int32_t out = 0;
    for(int j=0;j<HIDDEN2;++j) out += h2[j] * net.w3[j];
    return out / 1000; // scale
}

void nnue::load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&net), sizeof(Network));
}
