from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from surprise_travel.tools.custom_tool import MyCustomTool

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional

# trulens additions
from trulens.apps.custom import instrument


class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location of the activity")
    description: str = Field(..., description="Description of the activity")
    date: str = Field(..., description="Date of the activity")
    cuisine: str = Field(..., description="Cuisine of the restaurant")
    why_its_suitable: str = Field(
        ..., description="Why it's suitable for the traveler"
    )
    reviews: Optional[List[str]] = Field(..., description="List of reviews")
    rating: Optional[float] = Field(..., description="Rating of the activity")


class DayPlan(BaseModel):
    date: str = Field(..., description="Date of the day")
    activities: List[Activity] = Field(..., description="List of activities")
    restaurants: List[str] = Field(..., description="List of restaurants")
    flight: Optional[str] = Field(None, description="Flight information")


class Itinerary(BaseModel):
    name: str = Field(..., description="Name of the itinerary, something funny")
    day_plans: List[DayPlan] = Field(..., description="List of day plans")
    hotel: str = Field(..., description="Hotel information")


# Instrument some Crew class methods:
from crewai.crew import Crew
instrument.method(Crew, "kickoff")
instrument.method(Crew, "kickoff_async")
instrument.method(Crew, "train")
from crewai.agent import Agent
instrument.method(Agent, "execute_task")
from crewai.task import Task
instrument.method(Task, "execute_sync")
instrument.method(Task, "execute_async")
instrument.method(Task, "_execute_core")
from crewai.agents.crew_agent_executor import CrewAgentExecutor
instrument.method(CrewAgentExecutor, "invoke")

@CrewBase
class SurpriseTravelCrew:
    """SurpriseTravel crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def personalized_activity_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["personalized_activity_planner"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
            ],  # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def restaurant_scout(self) -> Agent:
        return Agent(
            config=self.agents_config["restaurant_scout"],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def itinerary_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config["itinerary_compiler"],
            tools=[SerperDevTool()],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def personalized_activity_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config["personalized_activity_planning_task"],
            agent=self.personalized_activity_planner(),
        )

    @task
    def restaurant_scenic_location_scout_task(self) -> Task:
        return Task(
            config=self.tasks_config["restaurant_scenic_location_scout_task"],
            agent=self.restaurant_scout(),
        )

    @task
    def itinerary_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config["itinerary_compilation_task"],
            agent=self.itinerary_compiler(),
            output_json=Itinerary,
        )

    @crew
    def create_crew(self) -> Crew:
        """Creates the SurpriseTravel crew"""
        # Important: Before wrapping this app with TruApp, assign the
        # result of this method to an attribute of this class as otherwise the
        # Crew class methods will not be instrumented.

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
