"""
场景：多人物品归属推理题
题目：有 4 个人：甲、乙、丙、丁，每人有且只有一种职业：教师、医生、工程师、律师。
已知条件：
1.甲不是教师，也不是律师。
2.乙不是医生。
3.如果丙是教师，那么丁是工程师。
4.工程师要么是丙，要么是丁。
5.丁不是律师。
请推理：甲、乙、丙、丁各自的职业是什么？
"""

from copy import deepcopy

#plan_and_solve
def plan_and_solve():
    """plan_and_solve解决职业归属问题"""

    people = ['甲', '乙', '丙', '丁']
    careers = ['教师', '医生', '工程师', '律师']
    print("==== plan_and_solve 推理过程 ====")
    #初始化数据
    possible ={
        '甲':careers.copy(),
        '乙':careers.copy(),
        '丙': careers.copy(),
        '丁': careers.copy()
    }
    #第一步确定甲的职业
    #根据条件1，排除教师和律师
    possible['甲'].remove('教师')
    possible['甲'].remove('律师')
    possible['乙'].remove('医生')
    possible['丁'].remove('律师')
    #根据条件4，排除工程师
    if '工程师' in possible['甲']:
        possible['甲'].remove('工程师')
    print(f"应用条件1后\n甲的可能职业：{possible['甲']}\n乙的可能职业：{possible['乙']}\n丙的可能职业：{possible['丙']}\n丁的可能职业：{possible['丁']}")
    if len(possible['甲']) == 1:
        jia_career = possible['甲'][0]
        print(f"应用条件3后，确定甲的职业为：{jia_career}")
    #第二步排除
    for i in people:
        if i != '甲'and '医生'in possible[i]:
            possible[i].remove('医生')
    print(f"排除医生后\n乙的可能职业：{possible['乙']}\n丙的可能职业：{possible['丙']}\n丁的可能职业：{possible['丁']}")
    #根据条件3确定丙和丁的职业
    print("假设丙是教师，那么丁是工程师，验证条件3")
    bing_career = '教师'
    ding_career = '工程师'
    possible['乙'].remove(bing_career)
    possible['乙'].remove(ding_career)
    if len(possible['乙']) == 1:
        possible['丙'] = bing_career
        possible['丁'] = ding_career
        yi_career = possible['乙'][0]
        print(f"验证通过\n甲的职业：{possible['甲'][0]}\n乙的职业：{yi_career}\n丙的职业：{possible['丙']}\n丁的职业：{possible['丁']}")
    return possible

#tree of thought
def is_valid(assignment,people,careers):
    """验证当前职业分配是否符合所有约束条件"""
    if assignment['甲'] is not None:
        if assignment['甲'] in ['教师','律师']:
            return False

    if assignment['乙'] is not None and assignment['甲'] == '医生':
        if assignment['乙'] == '医生':
            return False

    if assignment['丙'] == '教师' and assignment['丁'] is not None:
        if assignment['丁'] != '工程师':
            return False

    if assignment['丁'] is not None and assignment['丁'] == '律师':
        return False

    engineer_assigned = [p for p in people if assignment[p] == '工程师']
    if len(engineer_assigned) > 0:
        if engineer_assigned[0] not in ['丙','丁']:
            return False

    assigned_careers = [i for i in assignment.values() if i is not None]
    if len(assigned_careers) != len(set(assigned_careers)):
        return False

    return True

def generate_next_states(assignment,people,careers,current_person_idx):
    """生成当前节点的下一个分支状态"""
    if current_person_idx >= len(people):
        return []

    current_person = people[current_person_idx]

    if assignment[current_person] is not None:
        return generate_next_states(assignment,people,careers,current_person_idx+1)

    next_states = []

    for i in careers:
        new_assignment = deepcopy(assignment)
        new_assignment[current_person] = i

        if is_valid(new_assignment,people,careers):
            next_states.append(new_assignment)

    return next_states

def tot_search(people,careers,assignment=None,current_person_idx=0):
    """tree of thought 递归搜索"""

    if assignment is None:
        assignment = {p:None for p in people}

    if all(v is not None for v in assignment.values()):
        return [assignment]

    next_states = generate_next_states(assignment,people,careers,current_person_idx)

    solution = []
    for next_state in next_states:
        sub_solution = tot_search(people,careers,next_state,current_person_idx+1)
        solution.extend(sub_solution)

    return solution

def tree_of_thoughts():
    """主函数"""

    people = ['甲', '乙', '丙', '丁']
    careers = ['教师', '医生', '工程师', '律师']
    print("\n===== Tree of Thoughts 推理过程 =====")

    solution = tot_search(people,careers)
    print(f"\n找到{len(solution)}个合法解:")
    for i,sol in enumerate(solution,1):
        print(f"第{i}解:")
        for j,s in sol.items():
            print(f"{j}:{s}")

    best_assignment = solution[0] if solution else None
    if best_assignment:
        print("\n最优解:")
        for j,s in best_assignment.items():
            print(f"{j}:{s}")

    return best_assignment

ps_result = plan_and_solve()
tot_result = tree_of_thoughts()






