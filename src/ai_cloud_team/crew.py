from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env file
load_dotenv()

# Read the LLM choice from the environment variable
llm_choice = os.getenv("LLM_CHOICE", "llama").lower()

# Configure LLM based on choice
if llm_choice == "openai":
    from langchain.llms import OpenAI
    llm = OpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        temperature=0.7
    )
elif llm_choice == "llama":
    llm = LLM(
        model="ollama/llama3.1",
        base_url="http://localhost:11434"
    )
# Add more LLM options here as needed
else:
    raise ValueError(f"Unsupported LLM choice: {llm_choice}")

@CrewBase
class AiCloudTeamCrew():
	"""Ai Cloud Team crew"""

	@agent
	def cloud_architect(self) -> Agent:
		return Agent(
			config=self.agents_config['cloud_architect'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=llm
		)

	@agent
	def account_executive(self) -> Agent:
		return Agent(
			config=self.agents_config['account_executive'],
			verbose=True,
			llm=llm
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def estimation_task(self) -> Task:
		return Task(
			config=self.tasks_config['estimation_task'],
			# output_file='report.md'
		)

	@task
	def proposal_task(self) -> Task:
		return Task(
			config=self.tasks_config['proposal_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AiCloudTeam crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)