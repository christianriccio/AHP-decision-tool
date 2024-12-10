import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def geometric_mean(values):
    product = np.prod(values)
    return product ** (1 / len(values))

def calculate_priority_vector(matrix):
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    priority_vector = normalized_matrix.mean(axis=1)
    return priority_vector

def calculate_consistency_ratio(matrix, priority_vector):
    n = len(matrix)
    weighted_sum = matrix @ priority_vector
    lambda_max = (weighted_sum / priority_vector).mean()
    consistency_index = (lambda_max - n) / (n - 1)
    random_index = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    ri = random_index.get(n, 1.45)
    consistency_ratio = consistency_index / ri if n > 1 else 0
    return consistency_ratio

def create_pairwise_matrix(elements_list):
    n = len(elements_list)
    matrix = np.ones((n, n))
    return pd.DataFrame(matrix, index=elements_list, columns=elements_list)

def plot_radar_chart(criteria_list, data, labels):
    """
    criteria_list: list of criteria (axes of the radar)
    data: list of array with values for each criteria of each alternative
    labels: names of alternatives
    """
    N = len(criteria_list)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, d in enumerate(data):
        values = d.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=labels[i])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_list, fontsize=10)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

st.title("AHP Decision-Making Tool")
st.write("Welcome! This tool will help you through the Analytic Hierarchy Process (AHP).")

# Step 1: Define the objective
st.header("Step 1: Define the objective")
objective = st.text_input("Enter the objective of your decision (e.g., Selecting the best Data Lifecycle Model):")

# Step 2: Define the alternatives
st.header("Step 2: Define the alternatives")
st.write("Enter the alternatives to compare, separated by commas.")
alternatives = st.text_area("Alternatives (e.g., USGS, DataOne, Hindawi):")

if alternatives:
    alternatives_list = [alt.strip() for alt in alternatives.split(",")]
    st.write("Your alternatives are:")
    st.write(alternatives_list)

if st.button("Confirm Objective and Alternatives"):
    if not objective or not alternatives:
        st.error("Make sure you have entered both the objective and the alternatives!")
    else:
        st.success("Objective and alternatives saved! Let's move to the next step.")

# Step 3: Define the criteria
st.header("Step 3: Define the criteria")
criteria = st.text_area(
    "Enter the criteria to compare the alternatives, separated by commas (e.g., Governance, Privacy, Efficiency):")

if criteria:
    criteria_list = [crit.strip() for crit in criteria.split(",")]
    st.write("Your criteria are:")
    st.write(criteria_list)

if st.button("Confirm Criteria"):
    if not criteria:
        st.error("Please enter at least one criterion!")
    else:
        st.success("Criteria saved! Proceeding to the initial values matrix.")

# Step 4: Initial values matrix for alternatives based on criteria
st.header("Step 4: Initial Values Matrix for Alternatives Based on Criteria")
st.write("Enter a value (e.g., 1-10) for each Alternative-Criterion combination. These values are for reference.")
if criteria and alternatives:
    initial_values_matrix = pd.DataFrame(index=alternatives_list, columns=criteria_list, dtype=float)

    for alt in alternatives_list:
        st.subheader(f"Values for the alternative: {alt}")
        for c in criteria_list:
            val = st.number_input(f"Value of '{alt}' with respect to the criterion '{c}'",
                                  min_value=0.0, value=1.0, step=0.1,
                                  format="%.1f", key=f"initval-{alt}-{c}")
            initial_values_matrix.loc[alt, c] = val

    if st.button("Confirm Initial Matrix"):
        st.write("Initial Values Matrix:")
        st.dataframe(initial_values_matrix)
        st.success("Initial matrix saved! Moving to the pairwise comparisons for criteria.")

# Step 5: Pairwise comparisons for criteria
st.header("Step 5: Pairwise Comparisons for Criteria")
num_interviews = st.number_input("Number of interviews to conduct (N):", min_value=1, step=1, value=1)
st.write("Enter the pairwise comparison values for the criteria for each interview.")

if criteria:
    n_criteria = len(criteria_list)
    all_interview_matrices = []

    for interview_id in range(num_interviews):
        st.subheader(f"Interview {interview_id + 1}")
        comparison_matrix = create_pairwise_matrix(criteria_list)

        for i in range(n_criteria):
            for j in range(i + 1, n_criteria):
                value = st.number_input(
                    f"How important is '{criteria_list[i]}' compared to '{criteria_list[j]}'? (Interview {interview_id + 1})",
                    min_value=0.1, max_value=9.0, step=0.1, format="%.1f",
                    key=f"crit-{interview_id}-{i}-{j}"
                )
                comparison_matrix.iloc[i, j] = value
                comparison_matrix.iloc[j, i] = 1 / value

        st.write("Pairwise Comparison Matrix for Interview ", interview_id + 1, ":")
        st.dataframe(comparison_matrix)
        all_interview_matrices.append(comparison_matrix.values)

    if st.button("Calculate Final Criteria Matrix"):
        if len(all_interview_matrices) == 0:
            st.error("No interviews conducted!")
        else:
            all_interviews_array = np.array(all_interview_matrices)  # shape (N, n_criteria, n_criteria)

            # Calculate geometric mean for each element
            final_matrix = np.ones((n_criteria, n_criteria))
            for i in range(n_criteria):
                for j in range(n_criteria):
                    vals = all_interviews_array[:, i, j]
                    final_matrix[i, j] = geometric_mean(vals)

            final_criteria_matrix = pd.DataFrame(final_matrix, index=criteria_list, columns=criteria_list)
            st.write("Final Criteria Matrix (geometric mean of interviews):")
            st.dataframe(final_criteria_matrix)

            # Calculate priority vector and CR
            priority_vector = calculate_priority_vector(final_criteria_matrix)
            consistency_ratio = calculate_consistency_ratio(final_criteria_matrix.values, priority_vector)
            st.write("Priority Vector for Criteria:")
            for crit, p in zip(criteria_list, priority_vector):
                st.write(f"{crit}: {p:.4f}")
            st.write(f"Consistency Ratio (CR): {consistency_ratio:.4f}")
            if consistency_ratio > 0.1:
                st.warning("The criteria matrix is inconsistent! Review the comparisons.")
            else:
                st.success("The criteria matrix is consistent!")

            # Save criteria results
            with open("ahp_criteria_results.txt", "w") as f:
                f.write("Criteria Results (AHP):\n\n")
                f.write("Criteria and Priorities:\n")
                for criterion, pr in zip(criteria_list, priority_vector):
                    f.write(f"{criterion}: {pr:.4f}\n")
                f.write("\nConsistency Ratio (CR):\n")
                f.write(f"{consistency_ratio:.4f}\n")

            st.success("Criteria results saved in 'ahp_criteria_results.txt'.")

            # Calculate weighted values for each alternative (initial_values * priority_vector)
            # This shows how the initial values change based on criteria priorities.
            weighted_values_matrix = initial_values_matrix.copy()
            for i, c in enumerate(criteria_list):
                weighted_values_matrix[c] = weighted_values_matrix[c] * priority_vector[i]

            # Data preparation for radar plot
            # Each alternative will have one line. The axes are the criteria, and values are weighted.
            radar_data = [weighted_values_matrix.loc[alt].values for alt in alternatives_list]

            
            plot_radar_chart(criteria_list, radar_data, alternatives_list)
