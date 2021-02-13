student = [["Dave Jansz", "142757", "No Degree Declared", "1st Year"],
           [["CHEM201", "B"], ["MATH251", "A-"],
            ["PSYC205","B+"], ["STAT211", "B+"]]]

grades = {"A":4.0, "A-":3.7, "B+":3.3, "B":3.0, "B-":2.7, "C+":2.3, "C":2.0, "C-":1.7, "D+":1.3, "D":1.0, "F":0}

print("================Student enrollment record================")
print("Student name: %s" % student[0][0])
print("Student id: %s" % student[0][1])
print("Declared Major: %s" % student[0][2])

print("\nTranscript\n----------")

print("Course".ljust(10), "Grade".center(10), "    GPA")

gpa = 0.0
for courses in student[1]:
        print("%s %s %.2f" % (courses[0].ljust(12), courses[1].ljust(12),
                   grades[courses[1]]))
        gpa = gpa + grades[courses[1]]

print("\nStudent's overall GPA: %.2f" % (gpa/len(student[1])))
