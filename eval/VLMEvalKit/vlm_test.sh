
# Xattention
python run.py --data Video-MME_1fps --model Qwen2-VL-7B-Instruct --work-dir "./outputs_xattn_16_0.9" --reuse 
python run.py --data Video-MME_0.5fps_subs --model Qwen2-VL-7B-Instruct --work-dir "./outputs_xattn_16_0.9" --reuse 
# Baselines
python run.py --data Video-MME_1fps --model Qwen2-VL-7B-Instruct --work-dir "./outputs_flex" --reuse 
python run.py --data Video-MME_0.5fps_subs --model Qwen2-VL-7B-Instruct --work-dir "./outputs_flex" --reuse 
python run.py --data Video-MME_1fps --model Qwen2-VL-7B-Instruct --work-dir "./outputs_minfer" --reuse 
python run.py --data Video-MME_0.5fps_subs --model Qwen2-VL-7B-Instruct --work-dir "./outputs_minfer" --reuse 
python run.py --data Video-MME_1fps --model Qwen2-VL-7B-Instruct --work-dir "./outputs_full" --reuse 
python run.py --data Video-MME_0.5fps_subs --model Qwen2-VL-7B-Instruct --work-dir "./outputs_full" --reuse 