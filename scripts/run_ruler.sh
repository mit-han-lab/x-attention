cd eval/RULER/
bash setup.sh
cd scripts

./run.sh llama3.1-8b-chat synthetic  --stride 16  --metric xattn
./run.sh llama3.1-8b-chat synthetic  --stride 8  --metric xattn
./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric xattn
