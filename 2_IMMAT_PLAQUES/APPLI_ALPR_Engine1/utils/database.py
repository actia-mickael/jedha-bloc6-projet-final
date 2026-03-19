"""
SQLite Database Manager for ALPR Administration System.
Handles all database operations for residents and access logs.
"""

import sqlite3
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class DatabaseManager:
    """Manages SQLite database operations for ALPR system."""
    
    def __init__(self, db_path: str = "alpr.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_database(self):
        """Initialize database schema and import CSV if needed."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create residents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS residents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaque TEXT UNIQUE NOT NULL,
                nom TEXT NOT NULL,
                prenom TEXT NOT NULL,
                age INTEGER NOT NULL,
                telephone TEXT NOT NULL,
                adresse TEXT NOT NULL,
                ville TEXT NOT NULL,
                code_postal TEXT NOT NULL,
                abonnement TEXT NOT NULL CHECK(abonnement IN ('oui', 'non')),
                acces TEXT NOT NULL CHECK(acces IN ('oui', 'non'))
            )
        """)
        
        # Create logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaque TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                resultat TEXT NOT NULL CHECK(resultat IN ('autorisé', 'refusé')),
                normalized_plate TEXT NOT NULL
            )
        """)
        
        # Create index on logs for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_plaque ON logs(plaque)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC)
        """)
        
        conn.commit()
        
        # Check if residents table is empty and import CSV if needed
        cursor.execute("SELECT COUNT(*) FROM residents")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📦 Residents table is empty, importing from CSV...")
            self.import_from_csv()
        else:
            print(f"✅ Database initialized with {count} residents")
        
        conn.close()
    
    def import_from_csv(self, csv_path: str = "../plaques_avec_donnees.csv"):
        """
        Import residents from CSV file.
        
        Args:
            csv_path: Path to CSV file (relative to demo directory)
        """
        # Resolve path relative to this file's location
        base_dir = Path(__file__).parent.parent
        full_path = base_dir / csv_path
        
        if not full_path.exists():
            print(f"❌ CSV file not found: {full_path}")
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        imported = 0
        errors = 0
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cursor.execute("""
                            INSERT INTO residents (plaque, nom, prenom, age, telephone, 
                                                 adresse, ville, code_postal, abonnement, acces)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row['plaque'],
                            row['nom'],
                            row['prenom'],
                            int(row['age']),
                            row['telephone'],
                            row['adresse'],
                            row['ville'],
                            row['code_postal'],
                            row['abonnement'],
                            row['acces']
                        ))
                        imported += 1
                    except sqlite3.IntegrityError as e:
                        errors += 1
                        print(f"⚠️ Skipping duplicate plate: {row.get('plaque', 'unknown')}")
                    except Exception as e:
                        errors += 1
                        print(f"❌ Error importing row: {e}")
            
            conn.commit()
            print(f"✅ Imported {imported} residents from CSV ({errors} errors)")
        
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
        finally:
            conn.close()
    
    def get_all_residents(self) -> List[Dict]:
        """
        Get all residents from database.
        
        Returns:
            List of resident dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, plaque, nom, prenom, age, telephone, 
                   adresse, ville, code_postal, abonnement, acces
            FROM residents
            ORDER BY nom, prenom
        """)
        
        residents = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return residents
    
    def search_residents(self, query: str) -> List[Dict]:
        """
        Search residents by plate, nom, or prenom.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching resident dictionaries
        """
        if not query or not query.strip():
            return self.get_all_residents()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{query.strip()}%"
        
        cursor.execute("""
            SELECT id, plaque, nom, prenom, age, telephone, 
                   adresse, ville, code_postal, abonnement, acces
            FROM residents
            WHERE LOWER(plaque) LIKE LOWER(?)
               OR LOWER(nom) LIKE LOWER(?)
               OR LOWER(prenom) LIKE LOWER(?)
            ORDER BY nom, prenom
        """, (search_pattern, search_pattern, search_pattern))
        
        residents = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return residents
    
    def add_resident(self, data: Dict) -> Tuple[bool, str]:
        """
        Add a new resident to database.
        
        Args:
            data: Dictionary with resident data
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO residents (plaque, nom, prenom, age, telephone, 
                                     adresse, ville, code_postal, abonnement, acces)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['plaque'],
                data['nom'],
                data['prenom'],
                int(data['age']),
                data['telephone'],
                data['adresse'],
                data['ville'],
                data['code_postal'],
                data['abonnement'],
                data['acces']
            ))
            
            conn.commit()
            conn.close()
            return True, f"✅ Résident ajouté avec succès: {data['plaque']}"
        
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Erreur: La plaque {data['plaque']} existe déjà"
        
        except Exception as e:
            conn.close()
            return False, f"❌ Erreur lors de l'ajout: {str(e)}"
    
    def update_resident(self, resident_id: int, data: Dict) -> Tuple[bool, str]:
        """
        Update an existing resident.
        
        Args:
            resident_id: ID of resident to update
            data: Dictionary with updated data
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE residents
                SET plaque = ?, nom = ?, prenom = ?, age = ?, telephone = ?,
                    adresse = ?, ville = ?, code_postal = ?, abonnement = ?, acces = ?
                WHERE id = ?
            """, (
                data['plaque'],
                data['nom'],
                data['prenom'],
                int(data['age']),
                data['telephone'],
                data['adresse'],
                data['ville'],
                data['code_postal'],
                data['abonnement'],
                data['acces'],
                resident_id
            ))
            
            if cursor.rowcount == 0:
                conn.close()
                return False, f"❌ Résident avec ID {resident_id} non trouvé"
            
            conn.commit()
            conn.close()
            return True, f"✅ Résident mis à jour avec succès"
        
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Erreur: La plaque {data['plaque']} existe déjà"
        
        except Exception as e:
            conn.close()
            return False, f"❌ Erreur lors de la mise à jour: {str(e)}"
    
    def delete_resident(self, resident_id: int) -> Tuple[bool, str]:
        """
        Delete a resident from database.
        
        Args:
            resident_id: ID of resident to delete
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get plate before deleting for message
            cursor.execute("SELECT plaque FROM residents WHERE id = ?", (resident_id,))
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return False, f"❌ Résident avec ID {resident_id} non trouvé"
            
            plaque = row['plaque']
            
            cursor.execute("DELETE FROM residents WHERE id = ?", (resident_id,))
            conn.commit()
            conn.close()
            
            return True, f"✅ Résident supprimé: {plaque}"
        
        except Exception as e:
            conn.close()
            return False, f"❌ Erreur lors de la suppression: {str(e)}"
    
    def toggle_access(self, resident_id: int) -> Tuple[bool, str, str]:
        """
        Toggle access field for a resident.
        
        Args:
            resident_id: ID of resident
            
        Returns:
            Tuple of (success: bool, message: str, new_value: str)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current access value
            cursor.execute("SELECT acces, plaque FROM residents WHERE id = ?", (resident_id,))
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return False, f"❌ Résident avec ID {resident_id} non trouvé", ""
            
            current_access = row['acces']
            plaque = row['plaque']
            new_access = 'non' if current_access == 'oui' else 'oui'
            
            cursor.execute("UPDATE residents SET acces = ? WHERE id = ?", (new_access, resident_id))
            conn.commit()
            conn.close()
            
            status = "autorisé" if new_access == 'oui' else "bloqué"
            return True, f"✅ Accès {status} pour {plaque}", new_access
        
        except Exception as e:
            conn.close()
            return False, f"❌ Erreur: {str(e)}", ""
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about residents.
        
        Returns:
            Dictionary with statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total residents
        cursor.execute("SELECT COUNT(*) FROM residents")
        total = cursor.fetchone()[0]
        
        # Active (access granted)
        cursor.execute("SELECT COUNT(*) FROM residents WHERE acces = 'oui'")
        active = cursor.fetchone()[0]
        
        # Blocked (access denied)
        cursor.execute("SELECT COUNT(*) FROM residents WHERE acces = 'non'")
        blocked = cursor.fetchone()[0]
        
        # With subscription
        cursor.execute("SELECT COUNT(*) FROM residents WHERE abonnement = 'oui'")
        subscribed = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'active': active,
            'blocked': blocked,
            'subscribed': subscribed
        }
    
    def add_log(self, plaque: str, authorized: bool, normalized_plate: str = "") -> bool:
        """
        Add a detection log entry.
        
        Args:
            plaque: License plate text
            authorized: Whether access was granted
            normalized_plate: Normalized plate text
            
        Returns:
            Success status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            resultat = "autorisé" if authorized else "refusé"
            
            cursor.execute("""
                INSERT INTO logs (plaque, timestamp, resultat, normalized_plate)
                VALUES (?, ?, ?, ?)
            """, (plaque, timestamp, resultat, normalized_plate))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"❌ Error logging access: {e}")
            conn.close()
            return False
    
    def get_logs(self, plate_filter: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get access logs.
        
        Args:
            plate_filter: Optional filter by plate number
            limit: Maximum number of logs to return
            
        Returns:
            List of log dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if plate_filter:
            cursor.execute("""
                SELECT id, plaque, timestamp, resultat, normalized_plate
                FROM logs
                WHERE LOWER(plaque) LIKE LOWER(?)
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{plate_filter}%", limit))
        else:
            cursor.execute("""
                SELECT id, plaque, timestamp, resultat, normalized_plate
                FROM logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return logs
    
    def get_authorized_plates(self) -> List[str]:
        """
        Get list of all plates with access granted.
        
        Returns:
            List of plate numbers
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT plaque FROM residents WHERE acces = 'oui'")
        plates = [row['plaque'] for row in cursor.fetchall()]
        
        conn.close()
        return plates
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        exists = cursor.fetchone() is not None
        conn.close()
        
        return exists
