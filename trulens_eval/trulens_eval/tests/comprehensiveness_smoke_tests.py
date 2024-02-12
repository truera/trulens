# Author : Rajib
# Date: 02/11/2024
# This module is used to test the comprhensiveness feedback
import os

from dotenv import load_dotenv

from trulens_eval import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_provider = OpenAI()

score,reason = openai_provider.comprehensiveness_with_cot_reasons(source="The Taj Mahal is located on the right bank of "
                                                                     "the Yamuna River in a vast Mughal garden that "
                                                                     "encompasses nearly 17 hectares, in the Agra "
                                                                     "District in Uttar Pradesh. It was built by "
                                                                     "Mughal Emperor Shah Jahan in memory of his wife "
                                                                     "Mumtaz Mahal with construction starting in 1632 "
                                                                     "AD and completed in 1648 AD, with the mosque, "
                                                                     "the guest house and the main gateway on the "
                                                                     "south, the outer courtyard and its cloisters "
                                                                     "were added subsequently and completed in 1653 "
                                                                     "AD. The existence of several historical and "
                                                                     "Quaranic inscriptions in Arabic script have "
                                                                     "facilitated setting the chronology of Taj "
                                                                     "Mahal. For its construction, masons, "
                                                                     "stone-cutters, inlayers, carvers, painters, "
                                                                     "calligraphers, dome builders and other artisans "
                                                                     "were requisitioned from the whole of the empire "
                                                                     "and also from the Central Asia and Iran. "
                                                                     "Ustad-Ahmad Lahori was the main architect of "
                                                                     "the Taj Mahal.",summary="The Taj Mahal, "
                                                                                              "situated on the right "
                                                                                              "bank of the Yamuna "
                                                                                              "River in Agra "
                                                                                              "District, "
                                                                                              "Uttar Pradesh, "
                                                                                              "is a vast Mughal "
                                                                                              "garden covering "
                                                                                              "approximately 17 "
                                                                                              "hectares. Built by "
                                                                                              "Emperor Shah Jahan in "
                                                                                              "memory of his wife "
                                                                                              "Mumtaz Mahal, "
                                                                                              "construction began in "
                                                                                              "1632 AD and was "
                                                                                              "completed in 1648 AD. "
                                                                                              "Additional structures "
                                                                                              "like the mosque, "
                                                                                              "guest house, "
                                                                                              "main gateway, "
                                                                                              "outer courtyard, "
                                                                                              "and cloisters were "
                                                                                              "finished by 1653 AD. "
                                                                                              "The Taj Mahal's "
                                                                                              "chronology is aided by "
                                                                                              "historical and Quranic "
                                                                                              "inscriptions in Arabic "
                                                                                              "script. Skilled "
                                                                                              "artisans from across "
                                                                                              "the Mughal Empire, "
                                                                                              "Central Asia, "
                                                                                              "and Iran were employed "
                                                                                              "for its construction, "
                                                                                              "with Ustad-Ahmad "
                                                                                              "Lahori serving as the "
                                                                                              "main architect.")
print(score)
print(reason)