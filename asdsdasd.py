# import streamlit as st
# import jellyfish

# # def calculate_jaro_winkler_similarity(str1, str2):
# #     similarity = jellyfish.jaro_winkler_similarity(str1, str2)
# #     return similarity

# def calculate_jaro_similarity(str1, str2):
#     len1 = len(str1)
#     len2 = len(str2)

#     # Maximum allowed distance for matching characters
#     match_distance = max(len1, len2) // 2 - 1

#     # Initialize variables for matches, transpositions, and common characters
#     matches = 0
#     transpositions = 0
#     common_chars = []

#     # Find matching characters and transpositions
#     for i in range(len1):
#         start = max(0, i - match_distance)
#         end = min(i + match_distance + 1, len2)
#         for j in range(start, end):
#             if str1[i] == str2[j]:
#                 matches += 1
#                 if i != j:
#                     transpositions += 1
#                 common_chars.append(str1[i])
#                 break

#     # Calculate Jaro similarity
#     if matches == 0:
#         jaro_similarity = 0
#     else:
#         jaro_similarity = (
#             matches / len1 +
#             matches / len2 +
#             (matches - transpositions) / matches
#         ) / 3

#     return jaro_similarity

# def calculate_jaro_winkler_similarity(str1, str2, prefix_weight=0.1):
#     jaro_similarity = calculate_jaro_similarity(str1, str2)

#     # Calculate prefix match length
#     prefix_match_len = 0
#     for i in range(min(len(str1), len(str2))):
#         if str1[i] == str2[i]:
#             prefix_match_len += 1
#         else:
#             break

#     # Calculate Jaro-Winkler similarity
#     jaro_winkler_similarity = jaro_similarity + prefix_match_len * prefix_weight * (1 - jaro_similarity)

#     return jaro_winkler_similarity


# def main():
#     st.title("Jaro-Winkler Similarity")

#     # Query
#     query = st.text_input("Query:")

#     # Documents
#     num_documents = st.number_input("Number of Documents:", value=1, min_value=1, step=1)
#     documents = []
#     for i in range(num_documents):
#         document = st.text_input(f"Document {i+1}:")
#         documents.append(document)

#     # Calculate Jaro-Winkler Similarity for each document
#     similarities = []
#     for document in documents:
#         similarity = calculate_jaro_winkler_similarity(query, document)
#         similarities.append(similarity)

#     # Rank the documents based on similarity
#     ranked_documents = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

#     # Display the ranked documents
#     st.write("Ranked Documents:")
#     for rank, (document, similarity) in enumerate(ranked_documents, start=1):
#         st.write(f"Rank {rank}: Document: {document}, Similarity: {similarity}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import jellyfish
# import pandas as pd

# def calculate_jaro_winkler_similarity(str1, str2):
#     similarity = jellyfish.jaro_winkler_similarity(str1, str2)
#     return similarity

# def calculate_jaro_distance(str1, str2):
#     distance = 1 - calculate_jaro_winkler_similarity(str1, str2)
#     return distance

# def calculate_jaro_winkler_similarity_with_prefix(str1, str2, prefix_scale, prefix_length):
#     jaro_distance = calculate_jaro_distance(str1, str2)
#     jaro_winkler_similarity = jaro_distance + (prefix_scale * prefix_length * (1 - jaro_distance))
#     return jaro_winkler_similarity

# def main():
#     st.title("Jaro-Winkler Similarity")

#     # Documents
#     documents = st.text_area("Documents (One document per line):")
#     documents = documents.strip().split("\n")
#     D = len(documents) + 1

#     # Query
#     query = st.text_input("Query:")

#     # Prefix Scale and Length
#     prefix_scale = st.slider("Prefix Scale", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
#     prefix_length = st.slider("Prefix Length", min_value=0, max_value=10, value=0)

#     # Calculate Jaro Distance and Jaro-Winkler Similarity for each document
#     df_similarity = pd.DataFrame(index=['DJ', 'DW'], columns=[ 'D'+str(i) for i in range(1, D)])

#     for i in range(1, D):
#         dj = calculate_jaro_distance(query, documents[i-1])
#         dw = calculate_jaro_winkler_similarity_with_prefix(query, documents[i-1], prefix_scale, prefix_length)

#         df_similarity['D'+str(i)] = [dj, dw]

#         st.latex(r'''
#             \text{DJ}_{D''' + str(i) + r'''} = 1 - ''' + str(round(calculate_jaro_winkler_similarity(query, documents[i-1]), 4)) + r'''
#         ''')
#         st.latex(r'''
#             \text{DW}_{D''' + str(i) + r'''} = ''' + str(round(calculate_jaro_distance(query, documents[i-1]), 4)) + r''' + ''' + str(prefix_scale) + r''' \times ''' + str(prefix_length) + r''' \times (1 - ''' + str(round(calculate_jaro_distance(query, documents[i-1]), 4)) + r''')
#         ''')

#         st.write("DJ for Document", i, ":", dj)
#         st.write("DW for Document", i, ":", dw)

#     st.write("Similarity Matrix:")
#     st.dataframe(df_similarity)

# if __name__ == "__main__":
#     main()
# import streamlit as st
# import jellyfish
# import pandas as pd

# def calculate_jaro_winkler_similarity(str1, str2):
#     similarity = jellyfish.jaro_winkler_similarity(str1, str2)
#     return similarity

# def main():
#     st.title("Jaro-Winkler Similarity")

#     # Query
#     query = st.text_input("Query:")

#     # Documents
#     documents = st.text_area("Documents (One document per line):")
#     documents = documents.strip().split("\n")

#     # Adjust the length of query array to match the length of documents array
#     query = [query] * len(documents)

#     # Calculate Jaro-Winkler Similarity for each document
#     similarities = []
#     for document in documents:
#         similarity = calculate_jaro_winkler_similarity(query, document)
#         similarities.append(similarity)

#     # Rank the documents based on similarity
#     ranked_documents = sorted(range(len(documents)), key=lambda x: similarities[x], reverse=True)

#     # Create a DataFrame to display the ranked documents
#     df = pd.DataFrame({
#         "Rank": range(1, len(documents) + 1),
#         "Similarity": similarities,
#         "Document": documents
#     })

#     # Display the ranked documents
#     st.write("Ranked Documents:")
#     st.dataframe(df)

#     # Display the formulas
#     st.subheader("Formulas:")
#     dj_formula = f"D_j = \\frac{{m}}{{\\max({len(query[0])}, {len(documents[0])})}}"
#     dw_formula = f"D_w = D_j + (0.1 \\cdot (1 - D_j))"
#     st.latex(dj_formula.replace("m", str(len(query[0]))).replace("n", str(len(documents[0]))))
#     st.latex(dw_formula.replace("m", str(len(query[0]))).replace("n", str(len(documents[0]))))

# if __name__ == "__main__":
#     main()

import streamlit as st

def jaro_winkler(s1, s2):
  """Calculates the Jaro-Winkler distance between two strings."""
  m = _jaro_similarity(s1, s2)
  if m == 0:
    return 0
  t = _transpositions(s1, s2)
  p = _prefix_match(s1, s2)
  return m + (p / float(len(s1)) - 0.1) * (m - t)

def _jaro_similarity(s1, s2):
  """Calculates the Jaro similarity between two strings."""
  matches = 0
  transpositions = 0
  n = min(len(s1), len(s2))
  for i in range(n):
    if s1[i] == s2[i]:
      matches += 1
    elif i > 0 and s1[i] == s2[i - 1] and s1[i - 1] == s2[i]:
      transpositions += 1
  return float(matches + 2 * transpositions) / float(n + 1)

def _transpositions(s1, s2):
  """Counts the number of transpositions between two strings."""
  matches = 0
  transpositions = 0
  n = min(len(s1), len(s2))
  for i in range(n):
    if s1[i] == s2[i]:
      matches += 1
    elif matches > 0:
      matches -= 1
  return transpositions

def _prefix_match(s1, s2):
  """Returns the length of the longest common prefix between two strings."""
  i = 0
  j = 0
  while i < len(s1) and j < len(s2) and s1[i] == s2[j]:
    i += 1
    j += 1
  return i

def main():
  st.title("Jaro-Winkler Distance")
  s1 = st.text_input("String 1")
  s2 = st.text_input("String 2")
  dj = _jaro_similarity(s1, s2)
  dw = jaro_winkler(s1, s2)
  transpositions = _transpositions(s1, s2)
  prefix_match = _prefix_match(s1)

  # Display the Jaro and Jaro-Winkler distances in plain text.
  st.text(r"""
  \begin{align*}
    \text{DJ} &= \frac{m + 2t}{n + 1} \\
    \text{DW} &= \text{DJ} + \frac{p - t}{m}
  \end{align*}
  """.format(m=dj, n=len(s1), t=transpositions, p=prefix_match))

  # Display the actual values of the Jaro and Jaro-Winkler distances.
  st.write("DJ:", dj)
  st.write("DW:", dw)

if __name__ == "__main__":
  main()

