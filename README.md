# 🚀 Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following
[![Github](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=github&message=Rainier-rq/verl-if)]([https://github.com/Rainier-rq/verl-if])
![image](https://github.com/Rainier-rq/verl-if/blob/main/assets/11.png)
## 🔧 Environment Setup
- Python and required dependencies
  ```bash
   pip install -e .
   ```
- Recommended:
  - **1 single-GPU machine**: for RM service only
  - **Multi-node, multi-GPU systems**: for RL training (In our experiments, we used 3 nodes, each equipped with 8 A100 GPUs.)
---
## 1. Start RM Service
1. Open and edit `rm_service.py`:
   - Modify the **model_name** in the script to your trained reward model's path.
2. Run on development machine:
   ```bash
   python rm_service.py
   ```
3. After successful startup, the console will output the **IP address** and **port number** of the RM service.
---
## 2. Configure Reward Interface
1. Open `verl/utils/reward_score/instruction_reward.py`
2. Replace the **IP and Port** in the file with the address displayed when the RM service started.
3. If the model output format is different, adjust the following regex matching code:
   ```python
   answer_match = re.search(r"</think>\n\n(.*)", answer, re.DOTALL)
   ```
   Modify the `</think>\n\n(.*)` matching rule according to your model's output format to correctly extract the answer content.
---
## 3. RL Training 
1. Edit the training startup script:
   ```bash
   examples/qwen2_7b_instruction.sh
   ```
   - Replace the **model checkpoint path** with the actual path
2. Modify the configuration file:
   ```bash
   config.yaml
   ```
   - Configure the correct **model path**
   - Configure **number of nodes, communication ports** and other parameters
3. Start the Ray head node.
   ```bash
   ray start --include-dashboard=True --head --num-gpus 8 --max-worker-port 12800 --runtime-env-agent-port 20100 --dashboard-agent-grpc-port 20101 --dashboard-agent-listen-port 20102 --metrics-export-port 20103
   ```
4. Start the Ray worker node and connect to the head node.
   ```bash
   ray start --address $MASTER_IP:6379 --num-gpus 8 --max-worker-port 12800 --runtime-env-agent-port 20100 --dashboard-agent-grpc-port 20101 --dashboard-agent-listen-port 20102 --metrics-export-port 20103 --block
   ```
   - MASTER_IP refers to the IP address obtained in previous step.
7. Start training (execute on master node):
   ```bash
   sh examples/qwen2_7b_instruction.sh
   ```
---
## 4. Debugging and Notes
- If model output format is different, ensure **regex matching correctly extracts the answer**.
- For multi-machine deployment, pay attention to:
  - Whether paths are correct
  - Whether network connectivity is normal
  - Whether node count and port configurations are consistent across nodes
---
## 5. Usage Flow Overview
| Step               | Operation Content |
|--------------------|-------------------|
| **Start RM Service** | Run `python bia_use.py` on development machine and modify RM model path |
| **Configure Reward** | Fill in IP/Port in `instruction_reward.py` and adjust regex matching |
| **Configure Training Script** | Edit `qwen2_7b_instruction.sh` and `config.yaml`, update paths and node count |
| **Start Training**   | Execute `sh examples/qwen2_7b_instruction.sh` |
---
## Quick Command Summary
```bash
# 1. Start RM service (single-GPU development machine)
python bia_use.py  # Note: modify RM model path
# 2. Start multi-machine training (execute on master node)
sh examples/qwen2_7b_instruction.sh  # Note: configure paths and node parameters
```
---
## 🙋 Support
If you have any questions or issues, please feel free to raise them in Issues or contact us. 😄

---
## 📚 Acknowledgments
This project is built upon the excellent framework [EasyR1](https://github.com/hiyouga/EasyR1). We thank the contributors for their outstanding work.

---
### Thank you for using **verl-if** 🎉
This project welcomes everyone to provide opinions, suggestions, or submit PRs to improve it together!
