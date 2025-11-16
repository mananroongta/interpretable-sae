import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.trainer.agent_trainer import AgentTrainer

from HarmRLVR.agent import HarmRLVRAgent
from HarmRLVR.reward import harm_rlvr_reward

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("harmrlvr", "train")
    if train_dataset is None:
        raise ValueError("HarmRLVR train split is not registered. Run HarmRLVR/prepare_dataset.py first.")

    val_dataset = DatasetRegistry.load_dataset("harmrlvr", "val") or train_dataset

    env_args = {"reward_fn": harm_rlvr_reward}
    trainer = AgentTrainer(
        agent_class=HarmRLVRAgent,
        env_class=SingleTurnEnvironment,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
