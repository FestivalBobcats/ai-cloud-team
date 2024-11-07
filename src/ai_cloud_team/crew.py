from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from ai_cloud_team.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

llm = LLM(
	model="ollama/llama3.1",
	base_url="http://localhost:11434"
)


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