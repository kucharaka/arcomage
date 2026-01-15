#pragma once
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

enum class Res { Bricks, Gems, Beasts };

static inline std::string resName(Res r){
    if(r==Res::Bricks) return "Bricks";
    if(r==Res::Gems) return "Gems";
    return "Beasts";
}

struct Player {
    int tower=30, wall=10;
    int bricks=5, gems=5, beasts=5;
    int quarry=2, magic=2, zoo=2;

    void produce(){
        bricks += quarry;
        gems   += magic;
        beasts += zoo;
    }
};

struct Cost { Res type; int amount; };

enum class OpType {
    AddTower, AddWall,
    DamageOpp,
    AddResMe,
    AddGenMe,
    AddGenOpp,
    StealRes
};

struct Op {
    OpType type;
    Res res = Res::Bricks;
    int value = 0;
};

struct Card {
    std::string name;
    Cost cost;
    std::vector<Op> ops;
};

struct GameParams {
    int HAND_SIZE = 6;
    int WIN_TOWER = 100;
};

struct GameState {
    Player p[2];
    int current = 0;
    int turn = 0;
    bool done = false;
    int winner = -1;
};

class ArcomageEngine {
public:
    GameParams par;
    GameState st;

    std::vector<Card> pool;
    std::vector<int> deck;
    std::vector<int> hand[2];

    ArcomageEngine();

    // RL-friendly:
    // action: 0..H-1 play, H..2H-1 discard
    void reset(int seed = 0);
    void ensureTurnBegun();
    bool canPay(int who, const Card& c) const;

    int handSize()  const { return par.HAND_SIZE; }
    int actionSize() const { return 2 * par.HAND_SIZE; }
    int cardCount() const { return (int)pool.size(); }

    struct StepResult {
        double reward[2]{0,0};
        bool done=false;
        int winner=-1;
        std::string info;
    };
    StepResult step(int action);

    std::vector<float> getObservation(int me) const;
    std::vector<uint8_t> getActionMask(int me) const;

private:
    std::mt19937 rng;
    bool turnBegun = false;

    int& resRef(Player& p, Res r);
    int& genRef(Player& p, Res r);
    void dealDamage(Player& def, int dmg);

    void applyCardOps(int me, const Card& c);
    void checkTerminal();

    void buildCardPool();
    void buildDeck();
    int drawOne();
    void drawToHand(int who);
};
