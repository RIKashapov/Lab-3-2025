from clearml import Task

TASK_ID = "e0bc2b210afe4688bd493a8e0639ef56"

t = Task.get_task(task_id=TASK_ID)
hp = t.get_parameters()  # dict

print("TOTAL PARAMS:", len(hp))
# выведем только похожие на model / dataset / target
for k in sorted(hp.keys()):
    if "model" in k.lower() or "dataset" in k.lower() or "target" in k.lower():
        print(k, "=", hp[k])