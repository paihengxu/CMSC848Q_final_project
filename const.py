"""
Names, Races, Genders, Pronouns + Settings for GPT-3
Adapt for your own experiment!
"""
names = {"Black": {"man": ["Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie",
                           "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol"],
                   "woman": ["Latonya", "Tamika", "Ebony", "Latasha", "Keisha",
                             "Lillie", "Minnie", "Gwendolyn", "Bessie", "Marva"]},
         "White": {"man": ["Bradley", "Brett", "Scott", "Kurt", "Todd", "Chad",
                           "Matthew", "Dustin", "Shane", "Douglas"],
                   "woman": ["Beth", "Megan", "Kristin", "Jill", "Erin", "Colleen",
                             "Kathleen", "Heather", "Holly", "Laurie"]},
         "Asian": {"man": ["Viet", "Thong", "Qiang", "Kwok", "Hao", "Yang",
                           "Nam", "Huy", "Yuan", "Ho"],
                   "woman": ["Zhen", "Nga", "Lien", "Lam", "Hui", "Wing",
                             "Hoa", "Wai", "Min", "Huong"]},
         "Hispanic": {"man": ["Rigoberto", "Santos", "Javier", "Efrain", "Juan",
                              "Ramiro", "Jesus", "Humberto", "Gonzalo", "Hector"],
                      "woman": ["Guadalupe", "Marisela", "Guillermina", "Rocio",
                                "Yesenia", "Blanca", "Rosalba", "Elvia", "Alejandra", "Mayra"]}}

races = ['Black', 'White', 'Asian', 'Hispanic']
genders = ['man', 'woman']
medical_context_files = ['data_acute_cancer.csv', 'data_acute_non_cancer.csv', 'data_chronic_cancer.csv',
                         'data_chronic_non_cancer.csv', 'data_post_op.csv']
pronouns = {"subject": {"man": "he",
                        "woman": "she"},
            "possessive": {"man": "his",
                           "woman": "her"}}

OPTIONS_YESNO = ["Yes", "No"]
OPTIONS_DOSAGE = ["Low", "High", "nan"]

temp = 0.0
max_tokens = 150
logp = 5
stop = ["##"]

dose_low = "Dosage: Low (0.5 mg)"
dose_high = "Dosage: High (1 mg)"
