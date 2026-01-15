#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "engine.hpp"
#include "bot_onnx.hpp"

using namespace std;

static float lerp(float a, float b, float t) { return a + (b - a) * t; }
static float smoothTo(float current, float target, float speed, float dt){
    float t = 1.0f - std::exp(-speed * dt);
    return lerp(current, target, t);
}

static sf::Text makeText(const sf::Font& font, const string& s, unsigned size, sf::Color col){
    sf::Text t(font, s, size);
    t.setFillColor(col);
    return t;
}
static bool hit(const sf::FloatRect& r, sf::Vector2f p){ return r.contains(p); }

int main(){
    ArcomageEngine eng;
    eng.reset(123);

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u{1100, 700}), "Arcomage RL (P0 human vs P1 bot)");
    window.setFramerateLimit(60);

    sf::Font font;
    if(!font.openFromFile("assets/Roboto-Regular.ttf")){
        cerr << "Cannot load font assets/Roboto-Regular.ttf\n";
        return 1;
    }

    // bot
    int obsDim = (int)eng.getObservation(0).size();
    int actDim = eng.actionSize();
    BotONNX bot("models/policy.onnx", obsDim, actDim);

    float vTower[2]{(float)eng.st.p[0].tower, (float)eng.st.p[1].tower};
    float vWall[2] {(float)eng.st.p[0].wall,  (float)eng.st.p[1].wall};

    int selectedIdx = -1;
    string infoLine = "Your turn: click a card, then PLAY/DISCARD.";

    sf::Clock clock;

    while(window.isOpen()){
        float dt = clock.restart().asSeconds();

        // bot acts automatically when it's P1 turn
        if(!eng.st.done && eng.st.current == 1){
            eng.ensureTurnBegun();
            auto obs  = eng.getObservation(1);
            auto mask = eng.getActionMask(1);
            int action = bot.act(obs, mask, true); // stochastic
            auto sr = eng.step(action);
            infoLine = sr.info;
            selectedIdx = -1;
        }

        while (auto ev = window.pollEvent()) {
            if (ev->is<sf::Event::Closed>()) window.close();

            if (const auto* kp = ev->getIf<sf::Event::KeyPressed>()) {
                if (kp->code == sf::Keyboard::Key::Escape) window.close();
            }

            // clicks only on human turn
            if(!eng.st.done && eng.st.current == 0){
                if (const auto* mb = ev->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left) {
                        sf::Vector2f mp((float)mb->position.x, (float)mb->position.y);

                        eng.ensureTurnBegun();
                        int H = eng.par.HAND_SIZE;

                        sf::FloatRect playBtn({860.f, 560.f}, {200.f, 45.f});
                        sf::FloatRect discBtn({860.f, 610.f}, {200.f, 45.f});

                        if(hit(playBtn, mp) && selectedIdx >= 0){
                            auto sr = eng.step(selectedIdx);   // play
                            infoLine = sr.info;
                            selectedIdx = -1;
                        } else if(hit(discBtn, mp) && selectedIdx >= 0){
                            auto sr = eng.step(H + selectedIdx); // discard
                            infoLine = sr.info;
                            selectedIdx = -1;
                        } else {
                            float startX = 30, startY = 520;
                            float w = 130, h = 150, gap = 10;
                            for(int i=0;i<(int)eng.hand[0].size();++i){
                                sf::FloatRect cr({startX + i*(w+gap), startY}, {w, h});
                                if(hit(cr, mp)){ selectedIdx = i; break; }
                            }
                        }
                    }
                }
            }
        }

        for(int i=0;i<2;i++){
            vTower[i] = smoothTo(vTower[i], (float)eng.st.p[i].tower, 10.0f, dt);
            vWall[i]  = smoothTo(vWall[i],  (float)eng.st.p[i].wall,  10.0f, dt);
        }

        window.clear(sf::Color(25, 25, 35));

        // top info
        {
            string top = "Turn " + to_string(eng.st.turn) + " | Current: P" + to_string(eng.st.current) +
                         " | Win tower: " + to_string(eng.par.WIN_TOWER);
            auto t = makeText(font, top, 18, sf::Color(230,230,230));
            t.setPosition({20, 10});
            window.draw(t);
        }

        auto drawSide = [&](int i, float baseX){
            const float groundY = 480;
            const float scale = 3.0f;

            sf::RectangleShape wallRect;
            wallRect.setSize({80, max(0.f, vWall[i]) * scale});
            wallRect.setPosition({baseX, groundY - wallRect.getSize().y});
            wallRect.setFillColor(sf::Color(130,130,140));

            sf::RectangleShape towerRect;
            towerRect.setSize({80, max(0.f, vTower[i]) * scale});
            towerRect.setPosition({baseX + 110, groundY - towerRect.getSize().y});
            towerRect.setFillColor(sf::Color(200,170,120));

            window.draw(wallRect);
            window.draw(towerRect);

            const Player& p = eng.st.p[i];
            string s = "P"+to_string(i) +
                       "  T:"+to_string(p.tower)+" W:"+to_string(p.wall) +
                       "  B:"+to_string(p.bricks)+"(Q"+to_string(p.quarry)+")" +
                       "  G:"+to_string(p.gems)+"(M"+to_string(p.magic)+")" +
                       "  Be:"+to_string(p.beasts)+"(Z"+to_string(p.zoo)+")";
            auto txt = makeText(font, s, 16, sf::Color(220,220,220));
            txt.setPosition({baseX, 70});
            window.draw(txt);
        };

        drawSide(0, 80);
        drawSide(1, 650);

        {
            auto t = makeText(font, infoLine, 16, sf::Color(200,200,200));
            t.setPosition({20, 490});
            window.draw(t);
        }

        if(eng.st.done){
            sf::RectangleShape overlay({1100.f, 700.f});
            overlay.setFillColor(sf::Color(0,0,0,160));
            window.draw(overlay);

            string s = "Game Over. Winner: P" + to_string(eng.st.winner) + " (ESC to quit)";
            auto t = makeText(font, s, 28, sf::Color::White);
            auto b = t.getLocalBounds();
            t.setPosition({550 - b.size.x/2.f, 320 - b.size.y/2.f});
            window.draw(t);

            window.display();
            continue;
        }

        // hand of human only (P0)
        eng.ensureTurnBegun();

        float startX = 30, startY = 520;
        float w = 130, h = 150, gap = 10;

        for(int i=0;i<(int)eng.hand[0].size();++i){
            int cid = eng.hand[0][i];
            const Card& c = eng.pool[cid];

            sf::RectangleShape cardRect({w,h});
            cardRect.setPosition({startX + i*(w+gap), startY});

            bool payable = eng.canPay(0, c);
            cardRect.setFillColor(payable ? sf::Color(55, 70, 95) : sf::Color(45,45,45));

            if(i == selectedIdx){
                cardRect.setOutlineThickness(3.f);
                cardRect.setOutlineColor(sf::Color(240, 220, 120));
            } else {
                cardRect.setOutlineThickness(1.f);
                cardRect.setOutlineColor(sf::Color(90,90,90));
            }

            window.draw(cardRect);

            auto nameT = makeText(font, c.name, 14, sf::Color::White);
            nameT.setPosition({cardRect.getPosition().x + 6, cardRect.getPosition().y + 6});
            window.draw(nameT);

            string costS = "Cost: " + to_string(c.cost.amount) + " " + resName(c.cost.type);
            auto costT = makeText(font, costS, 14, sf::Color(220,220,220));
            costT.setPosition({cardRect.getPosition().x + 6, cardRect.getPosition().y + 110});
            window.draw(costT);
        }

        // buttons
        sf::FloatRect playBtn({860.f, 560.f}, {200.f, 45.f});
        sf::FloatRect discBtn({860.f, 610.f}, {200.f, 45.f});

        auto drawBtn = [&](sf::FloatRect r, const string& s, sf::Color col){
            sf::RectangleShape b({r.size.x, r.size.y});
            b.setPosition({r.position.x, r.position.y});
            b.setFillColor(col);
            b.setOutlineThickness(2);
            b.setOutlineColor(sf::Color(20,20,20));
            window.draw(b);

            auto t = makeText(font, s, 18, sf::Color::Black);
            auto lb = t.getLocalBounds();
            t.setPosition({r.position.x + (r.size.x-lb.size.x)/2.f, r.position.y + (r.size.y-lb.size.y)/2.f - 6});
            window.draw(t);
        };

        drawBtn(playBtn, "PLAY", sf::Color(130, 210, 130));
        drawBtn(discBtn, "DISCARD", sf::Color(210, 130, 130));

        window.display();
    }

    return 0;
}
