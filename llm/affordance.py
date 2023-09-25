from env.env import PLACE_TARGETS

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 0.2
      continue
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores


def affordance_score2(task, found_objects):
  block_name="box"
  bowl_name="circle"
  pick, place = task.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  if pick in found_objects and place in found_objects: return 1
  elif pick == 'done': return 2
  else: return 0
