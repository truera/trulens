#!/usr/bin/env python
import sys
from surprise_travel.crew import SurpriseTravelCrew

surprise_travel_crew = SurpriseTravelCrew()

from trulens.apps.custom import TruCustomApp
from trulens.core import TruSession

session = TruSession()

session.reset_database()

tru_suprise_travel_crew = TruCustomApp(
    surprise_travel_crew,
    app_name="SurpriseTravelCrew",
    app_version="1.0.0",
    feedbacks=[],
)


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'origin': 'São Paulo, GRU',
        'destination': 'New York, JFK',
        'age': 31,
        'hotel_location': 'Brooklyn',
        'flight_information': 'GOL 1234, leaving at June 30th, 2024, 10:00',
        'trip_duration': '14 days'
    }
    with tru_suprise_travel_crew as recorder:
        result = SurpriseTravelCrew().crew().kickoff(inputs=inputs)
    print(result)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'origin': 'São Paulo, GRU',
        'destination': 'New York, JFK',
        'age': 31,
        'hotel_location': 'Brooklyn',
        'flight_information': 'GOL 1234, leaving at June 30th, 2024, 10:00',
        'trip_duration': '14 days'
    }
    try:
        surprise_travel_crew.crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
