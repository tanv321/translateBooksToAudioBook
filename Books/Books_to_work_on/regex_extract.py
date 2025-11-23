import re

with open('gaza_an_inquest_into_its_martyrdom_cleaned2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Your regex pattern
pattern = r'(["' + "'" + r';!?]|(?<!\d)\.)(\d+)'
matches = re.findall(pattern, text)

# Write to output file
with open('matches.txt', 'w', encoding='utf-8') as f:
    for match in matches:
        f.write(f'{match[0]}{match[1]}\n')

print(f"Found {len(matches)} matches. Saved to matches.txt")

""" The script finds what regex pattern we provided and then it will show you in the matches text file what was captured."""