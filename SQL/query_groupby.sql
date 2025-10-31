SELECT age, COUNT(rowid)
FROM students
GROUP BY age
HAVING COUNT(rowid) > 1;