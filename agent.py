from planner import Planner
from tools import calculate


class Agent:

    def __init__(self, api_key):
        self.planner = Planner(api_key)

    def run(self, query):

        decision = self.planner.decide(query)

        print("Planner Decision:")
        print(decision)

        lines = decision.splitlines()
        tool = lines[0].split(":")[1].strip()
        tool_input = lines[1].split(":")[1].strip()
        if tool == "calculate":
            result = calculate(tool_input)
            answer = self.planner.respond(
                query,
                result
            )
            return answer
        else:
            return "No tool used."

        
