import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.database import DatabaseManager

def test_db():
    # Use a test db file to avoid messing with real one if it exists, 
    # but we want to test CSV import which happens on init.
    # So let's use a temp db name.
    db_name = "test_alpr.db"
    
    if os.path.exists(db_name):
        os.remove(db_name)

    print("🧪 Testing Database Manager...")
    db = DatabaseManager(db_name)

    # Test 1: Import
    residents = db.get_all_residents()
    print(f"Residents count: {len(residents)}")
    
    # Check if CSV import worked (assuming csv exists in parent dir)
    if len(residents) == 0:
        print("⚠️ Warning: No residents imported (CSV might be missing in test env), adding manually for tests")
    else:
        print("✅ CSV Import verified")

    # Test 2: Add
    new_res = {
        'plaque': 'TEST-999',
        'nom': 'Tester',
        'prenom': 'Joe',
        'age': 40,
        'telephone': '0102030405',
        'adresse': 'Test St',
        'ville': 'Test City',
        'code_postal': '00000',
        'abonnement': 'non',
        'acces': 'non'
    }
    success, msg = db.add_resident(new_res)
    print(f"Add resident: {msg}")
    assert success, "Add resident failed"

    # Test 3: Search
    results = db.search_residents("TEST-999")
    print(f"Search 'TEST-999' found: {len(results)}")
    assert len(results) == 1, "Search failed"
    res_id = results[0]['id']

    # Test 4: Toggle Access
    success, msg, status = db.toggle_access(res_id)
    print(f"Toggle access: {msg} -> {status}")
    assert status == 'oui', "Toggle access failed"
    
    # Check stats
    stats = db.get_statistics()
    print(f"Stats: {stats}")
    assert stats['active'] >= 1

    # Test 5: Delete
    success, msg = db.delete_resident(res_id)
    print(f"Delete resident: {msg}")
    assert success, "Delete failed"
    
    results = db.search_residents("TEST-999")
    assert len(results) == 0, "Delete verification failed"

    # Test 6: Logs
    db.add_log("TEST-LOG", True, "TESTLOG")
    logs = db.get_logs()
    print(f"Logs count: {len(logs)}")
    assert len(logs) > 0, "Logging failed"

    print("✅ All backend tests passed!")
    
    # Cleanup
    if os.path.exists(db_name):
        os.remove(db_name)

if __name__ == "__main__":
    test_db()
