#include "engine.hpp"

ArcomageEngine::ArcomageEngine()
: rng((uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count())
{
    buildCardPool();
    reset(123);
}

int& ArcomageEngine::resRef(Player& p, Res r){
    if(r==Res::Bricks) return p.bricks;
    if(r==Res::Gems) return p.gems;
    return p.beasts;
}
int& ArcomageEngine::genRef(Player& p, Res r){
    if(r==Res::Bricks) return p.quarry;
    if(r==Res::Gems) return p.magic;
    return p.zoo;
}

void ArcomageEngine::dealDamage(Player& def, int dmg){
    int w = std::min(def.wall, dmg);
    def.wall -= w;
    dmg -= w;
    if(dmg>0) def.tower -= dmg;
}

void ArcomageEngine::buildCardPool(){
    pool.clear();
    pool.push_back({"Wall +6", {Res::Bricks,4}, {{OpType::AddWall, Res::Bricks, 6}}});
    pool.push_back({"Tower +5", {Res::Bricks,5}, {{OpType::AddTower, Res::Bricks, 5}}});
    pool.push_back({"Quarry +1", {Res::Bricks,8}, {{OpType::AddGenMe, Res::Bricks, 1}}});

    pool.push_back({"Magic +1", {Res::Gems,8}, {{OpType::AddGenMe, Res::Gems, 1}}});
    pool.push_back({"Zoo +1", {Res::Beasts,8}, {{OpType::AddGenMe, Res::Beasts, 1}}});

    pool.push_back({"Fireball (7 dmg)", {Res::Gems,6}, {{OpType::DamageOpp, Res::Gems, 7}}});
    pool.push_back({"Lightning (10 dmg)", {Res::Gems,9}, {{OpType::DamageOpp, Res::Gems, 10}}});
    pool.push_back({"Drain 4 gems", {Res::Gems,4}, {{OpType::StealRes, Res::Gems, 4}}});

    pool.push_back({"Raid (6 dmg, +2 beasts)", {Res::Beasts,5},
                    {{OpType::DamageOpp, Res::Beasts, 6}, {OpType::AddResMe, Res::Beasts, 2}}});

    pool.push_back({"Sabotage (-1 opp quarry)", {Res::Beasts,7}, {{OpType::AddGenOpp, Res::Bricks, -1}}});

    pool.push_back({"Rebuild (+3 wall, +3 tower)", {Res::Bricks,6},
                    {{OpType::AddWall, Res::Bricks, 3}, {OpType::AddTower, Res::Bricks, 3}}});
}

void ArcomageEngine::buildDeck(){
    deck.clear();
    for(int i=0;i<(int)pool.size();++i)
        for(int k=0;k<4;k++) deck.push_back(i);
    std::shuffle(deck.begin(), deck.end(), rng);
}

int ArcomageEngine::drawOne(){
    if(deck.empty()) buildDeck();
    int cid = deck.back();
    deck.pop_back();
    return cid;
}

void ArcomageEngine::drawToHand(int who){
    while((int)hand[who].size() < par.HAND_SIZE){
        hand[who].push_back(drawOne());
    }
}

void ArcomageEngine::reset(int seed){
    rng.seed((uint32_t)seed);

    st = GameState{};
    st.p[0] = Player{};
    st.p[1] = Player{};
    st.current = 0;
    st.turn = 0;
    st.done = false;
    st.winner = -1;

    buildDeck();
    hand[0].clear();
    hand[1].clear();
    drawToHand(0);
    drawToHand(1);

    turnBegun = false;
}

void ArcomageEngine::ensureTurnBegun(){
    if(turnBegun || st.done) return;
    int me = st.current;
    st.p[me].produce();
    drawToHand(me);
    turnBegun = true;
}

bool ArcomageEngine::canPay(int who, const Card& c) const {
    const Player& p = st.p[who];
    int have = (c.cost.type==Res::Bricks)?p.bricks:(c.cost.type==Res::Gems)?p.gems:p.beasts;
    return have >= c.cost.amount;
}

void ArcomageEngine::applyCardOps(int me, const Card& c){
    int opp = 1-me;
    for(const auto& op : c.ops){
        switch(op.type){
            case OpType::AddTower: st.p[me].tower += op.value; break;
            case OpType::AddWall:  st.p[me].wall  += op.value; break;
            case OpType::DamageOpp: dealDamage(st.p[opp], op.value); break;
            case OpType::AddResMe:  resRef(st.p[me], op.res) += op.value; break;
            case OpType::AddGenMe:  genRef(st.p[me], op.res) = std::max(0, genRef(st.p[me], op.res) + op.value); break;
            case OpType::AddGenOpp: genRef(st.p[opp], op.res) = std::max(0, genRef(st.p[opp], op.res) + op.value); break;
            case OpType::StealRes: {
                int take = std::min(op.value, resRef(st.p[opp], op.res));
                resRef(st.p[opp], op.res) -= take;
                resRef(st.p[me], op.res) += take;
            } break;
        }
    }
}

void ArcomageEngine::checkTerminal(){
    for(int i=0;i<2;i++){
        if(st.p[i].tower >= par.WIN_TOWER){ st.done=true; st.winner=i; return; }
    }
    for(int i=0;i<2;i++){
        if(st.p[i].tower <= 0){ st.done=true; st.winner=1-i; return; }
    }
}

ArcomageEngine::StepResult ArcomageEngine::step(int action){
    StepResult sr;

    if(st.done){
        sr.done = true;
        sr.winner = st.winner;
        sr.info = "Game already finished";
        return sr;
    }

    ensureTurnBegun();

    int me = st.current;
    int opp = 1-me;

    int H = par.HAND_SIZE;
    bool discard = (action >= H);
    int idx = discard ? (action - H) : action;

    auto invalid = [&](const std::string& msg){
        sr.reward[me] = -0.05;
        sr.reward[opp] = +0.05;
        sr.info = "Invalid: " + msg;
    };

    if(idx < 0 || idx >= (int)hand[me].size()){
        invalid("index out of range");
    } else {
        int cid = hand[me][idx];
        const Card& c = pool[cid];

        if(!discard){
            if(!canPay(me,c)){
                invalid("cannot pay");
            } else {
                resRef(st.p[me], c.cost.type) -= c.cost.amount;
                applyCardOps(me, c);
                sr.info = "P" + std::to_string(me) + " played: " + c.name;
                hand[me].erase(hand[me].begin()+idx);
            }
        } else {
            sr.info = "P" + std::to_string(me) + " discarded: " + c.name;
            hand[me].erase(hand[me].begin()+idx);
        }
    }

    checkTerminal();
    if(st.done){
        sr.reward[st.winner] = +1.0;
        sr.reward[1-st.winner] = -1.0;
        sr.done = true;
        sr.winner = st.winner;
        return sr;
    }

    turnBegun = false;
    st.current = 1 - st.current;
    st.turn++;

    return sr;
}

std::vector<uint8_t> ArcomageEngine::getActionMask(int me) const {
    int H = par.HAND_SIZE;
    std::vector<uint8_t> m(2*H, 0);

    for(int i=0;i<H;i++){
        int cid = hand[me][i];
        m[i] = canPay(me, pool[cid]) ? 1 : 0; // play
        m[H+i] = 1;                           // discard
    }
    return m;
}

std::vector<float> ArcomageEngine::getObservation(int me) const {
    int H = par.HAND_SIZE;
    int N = (int)pool.size();
    std::vector<float> obs;
    obs.reserve(18 + H*(N+1));

    auto pushPlayer = [&](const Player& p, int handSz){
        obs.push_back((float)p.tower);
        obs.push_back((float)p.wall);
        obs.push_back((float)p.bricks);
        obs.push_back((float)p.gems);
        obs.push_back((float)p.beasts);
        obs.push_back((float)p.quarry);
        obs.push_back((float)p.magic);
        obs.push_back((float)p.zoo);
        obs.push_back((float)handSz);
    };

    pushPlayer(st.p[me], (int)hand[me].size());
    pushPlayer(st.p[1-me], (int)hand[1-me].size());

    for(int i=0;i<H;i++){
        int cid = hand[me][i];
        for(int k=0;k<N;k++) obs.push_back(k==cid ? 1.f : 0.f);
        obs.push_back(canPay(me, pool[cid]) ? 1.f : 0.f);
    }
    return obs;
}
