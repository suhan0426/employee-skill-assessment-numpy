import numpy as np

np.set_printoptions(precision=2, suppress=True)

employees = np.array(["E01", "E02", "E03", "E04",
                      "E05", "E06", "E07", "E08"])
roles = np.array(["Data Analyst", "ML Engineer", "Technical Trainer"])
skill_names = np.array(["Python", "SQL", "Machine Learning", "Communication"])

# Task 1: Employee Skill Data Management
print("\n========== TASK 1: EMPLOYEE SKILL DATA ==========")
skills = np.array([
    [85, 78, 92, 70],
    [72, 90, 65, 85],
    [95, 68, 88, 75],
    [60, 85, 55, 90],
    [88, 92, 80, 65],
    [75, 70, 95, 80],
    [68, 88, 72, 95],
    [90, 82, 85, 88]
])

print("Employee Skill Matrix:\n", skills)
print("Shape:", skills.shape)
print("Number of employees:", skills.shape[0])
print("Number of skills:", skills.shape[1])

flat = skills.flatten()
print("Flattened array:\n", flat)
reshaped = flat.reshape(8, 4)
print("Reshaped matrix:\n", reshaped)
print("E01 Python score:", skills[0, 0])
print("E08 Communication score:", skills[7, 3])

# Task 2: Job Role Suitability Analysis
print("\n========== TASK 2: ROLE SUITABILITY ==========")
weights = np.array([
    [0.30, 0.35, 0.20],
    [0.40, 0.10, 0.15],
    [0.20, 0.45, 0.25],
    [0.10, 0.10, 0.40]
])
print("Role Weight Matrix:\n", weights)
suitability = np.matmul(skills, weights)
print("Suitability Matrix:\n", suitability)

# Task 3: Performance-Based Bonus System
print("\n========== TASK 3: BONUS SYSTEM ==========")
bonus = np.array([3, 5, 2])
updated_scores = suitability + bonus
print("Original suitability scores:\n", suitability)
print("Bonus array:", bonus)
print("Updated scores after broadcasting:\n", updated_scores)

# Task 4: Skill Improvement Simulation
print("\n========== TASK 4: TRAINING SIMULATION ==========")
improvement = np.array([0, 0, 10, 0])
trained_skills = np.clip(skills + improvement, 0, 100)
trained_suitability = np.matmul(trained_skills, weights)
print("Training improvement:", improvement)
print("Trained skill matrix:\n", trained_skills)
print("Suitability after training:\n", trained_suitability)
print("Suitability improvement:\n", trained_suitability - suitability)

# Task 5: Random Performance Simulation
print("\n========== TASK 5: RANDOM PERFORMANCE ==========")
np.random.seed(42)
random_points = np.random.rand(8, 4) * 5
random_skills = np.clip(skills + random_points, 0, 100)
random_suitability = np.matmul(random_skills, weights)
print("Random additional points:\n", random_points)
print("Skills after positive random points:\n", random_skills)
print("Suitability after positive simulation:\n", random_suitability)

random_fluctuations = np.random.randn(8, 4) * 2
fluctuated_skills = np.clip(skills + random_fluctuations, 0, 100)
fluctuated_suitability = np.matmul(fluctuated_skills, weights)
print("Positive/negative fluctuations:\n", random_fluctuations)
print("Suitability after fluctuation simulation:\n", fluctuated_suitability)

# Task 6: Individual Employee Evaluation
print("\n========== TASK 6: E03 EVALUATION ==========")
e03_skills = skills[2]
ml_engineer_weights = weights[:, 1]
e03_ml_score = np.dot(e03_skills, ml_engineer_weights)
print("E03 skill scores:", e03_skills)
print("ML Engineer weight vector:", ml_engineer_weights)
print("E03 ML Engineer suitability (np.dot):", e03_ml_score)
print("Matrix multiplication verification:", suitability[2, 1])

# Task 7: Recommendations and Business Insights
print("\n========== TASK 7: BUSINESS INSIGHTS ==========")
for role_index in range(len(roles)):
    best_index = np.argmax(suitability[:, role_index])
    print("Highest for", roles[role_index], ":",
          employees[best_index], "Score:", suitability[best_index, role_index])

for employee_index in range(len(employees)):
    best_role_index = np.argmax(suitability[employee_index])
    tied_roles = roles[suitability[employee_index] == np.max(suitability[employee_index])]
    print(employees[employee_index], "Best role(s):", ", ".join(tied_roles))

average_skills = np.mean(skills, axis=0)
lowest_skill_index = np.argmin(average_skills)
print("Average skill scores:", average_skills)
print("Skill needing greatest improvement:", skill_names[lowest_skill_index])

qualified = np.any(suitability >= 85, axis=1)
print("Employees scoring at least 85 in any role:", np.sum(qualified))
print("Qualified employees:", employees[qualified])

role_improvement = np.mean(trained_suitability - suitability, axis=0)
best_improved_role = np.argmax(role_improvement)
print("Average role improvement after ML training:", role_improvement)
print("Role benefiting most from ML training:", roles[best_improved_role])

print("\n========== PROJECT COMPLETED ==========")
