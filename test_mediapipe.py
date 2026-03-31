import mediapipe as mp

print("Path:", mp.__file__)
print("Has solutions:", hasattr(mp, "solutions"))

mp_face_mesh = mp.solutions.face_mesh
print("FaceMesh OK")
