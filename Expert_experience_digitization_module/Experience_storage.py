import os

def practice_exp_storage(goal:str,last_response:str,response:str,expert_experience_path:str):
    if not os.path.exists(expert_experience_path):
        with open(expert_experience_path, 'w') as file:
            if last_response=="":
                context = "goal: " + goal + "\n"+ "Your response: " + response +"\n" + "**\n"
            else:
                context = "goal: " + goal + "\n" + "Latest Feedback:" + last_response + "\n" + "Your response: " + response + "\n" + "**\n"
            file.write(context)
    else:
        with open(expert_experience_path, 'a') as file:
            if last_response=="":
                context = "goal: " + goal + "\n"+ "Your response: " + response +"\n" + "**\n"
            else:
                context = "goal: " + goal + "\n" + "Latest Feedback:" + last_response + "\n" + "Your response: " + response + "\n" + "**\n"
            file.write(context)


def expert_guidance_exp_storage(goal:str,last_response:str,command:str,response:str,expert_experience_path:str):
    if not os.path.exists(expert_experience_path):
        with open(expert_experience_path, 'w') as file:
            if last_response=="":
                context = "goal: " + goal + "\n" + "COB command: " + command +"\n" + "Your response: " + "Your response:" + response +"\n" + "**\n"
            else:
                context = "goal: " + goal + "\n" + "Latest Feedback:" + last_response + "\n" + "COB command: " + command + "\n" + "Your response: " + "Your response:" + response + "\n" + "**\n"
            file.write(context)
    else:
        with open(expert_experience_path, 'a') as file:
            if last_response=="":
                context = "goal: " + goal + "\n" + "COB command: " + command +"\n" + "Your response: " + "Your response:" + response +"\n" + "**\n"
            else:
                context = "goal: " + goal + "\n" + "Latest Feedback:" + last_response + "\n" + "COB command: " + command + "\n" + "Your response: " + "Your response:" + response + "\n" + "**\n"
            file.write(context)