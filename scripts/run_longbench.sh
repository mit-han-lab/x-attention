models="Meta-Llama-3.1-8B-Instruct"
# Xattention
methods="xattn"
# Baselines
# methods="full flex minference"
tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for method in $methods; do
            bash scripts/longbench.sh $model $task $method
        done
    done
done

cd eval/LongBench
for model in $models; do
    python -u eval.py --model $model &
done
