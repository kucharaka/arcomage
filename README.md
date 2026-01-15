Сборка:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

Обучение:
PYTHONPATH=build python3 python/train_selfplay.py

Запуск игры:
./build/arcomage_game
