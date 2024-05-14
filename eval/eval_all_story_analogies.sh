echo "Starting llama3"
python3 eval/eval_story_analogy.py --model llama3 --prompt basic_prompt.txt
python3 eval/eval_story_analogy.py --model llama3 --prompt cot.txt
python3 eval/eval_story_analogy.py --model llama3 --prompt cot_structured.txt
echo "Starting mistral7b"
python3 eval/eval_story_analogy.py --model mistral7b --prompt basic_prompt.txt
python3 eval/eval_story_analogy.py --model mistral7b --prompt cot.txt
python3 eval/eval_story_analogy.py --model mistral7b --prompt cot_structured.txt
echo "Starting starling7b-beta"
python3 eval/eval_story_analogy.py --model starling7b-beta --prompt basic_prompt.txt
python3 eval/eval_story_analogy.py --model starling7b-beta --prompt cot.txt
python3 eval/eval_story_analogy.py --model starling7b-beta --prompt cot_structured.txt
