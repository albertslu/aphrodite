import json

# Load profiles
with open("extracted_500_random_profiles.json", "r") as f:
    profiles = json.load(f)

# Count matches
matches = []
for profile in profiles:
    # Check basic criteria
    if (
        profile.get("sex") == "m" and  # Male
        profile.get("ethnicity", "").lower() in ["asian", "white"] and  # Asian or White
        19 <= profile.get("age", 0) <= 27  # Age 21-25 ±2 years
    ):
        matches.append({
            "age": profile.get("age"),
            "ethnicity": profile.get("ethnicity"),
            "location": profile.get("location"),
            "essays": [profile.get(f"essay{i}", "") for i in range(10)]
        })

print(f"Found {len(matches)} profiles matching basic criteria:")
print("\nBreakdown:")
ethnicities = {}
for match in matches:
    ethnicity = match.get("ethnicity", "N/A")
    ethnicities[ethnicity] = ethnicities.get(ethnicity, 0) + 1

for ethnicity, count in ethnicities.items():
    print(f"{ethnicity}: {count} profiles")

# Save matches to file for inspection
with open("matching_profiles.json", "w") as f:
    json.dump(matches, f, indent=2)
