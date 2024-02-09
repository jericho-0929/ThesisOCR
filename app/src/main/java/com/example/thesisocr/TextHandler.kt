package com.example.thesisocr

import android.util.Log

// TODO: Test the TextHandler class.
// TODO: Implement NeuralModel.kt for handling the neural network model.

/**
 * TextHandler class for processing extracted text
 */

/**
 * Neural Network outputs are to be added to a mutable map called entriesMap.
 * Keys are the identity card number.
 * Value is a list of strings containing
 * the last name, given name, middle name, date of birth, and address.
 */

/**
 * Format of National Identity Card:
 * 1. Last Name (String)
 * 2. Given Name (String)
 * 3. Middle Name (String)
 * 3. Identity Number (xxxx-xxxx-xxxx) (String)
 * 4. Date of Birth (MMMM-DD-YYYY) (String)
 * 5. Address (String)
 */

/**
 * Methods for determining the type of string extracted from the image.
 * Identity Number and Date of Birth uses pattern-matching.
 * Explore use of Named Entity Recognition (NER) for extracting names information.
 * Explore use of regular expressions for extracting address information.
 */
class TextHandler {
    // Data class for storing entry details
    data class entryDetails(var typeOfEntry: String, var entry: String)
    // Public variables
    var keyEntries = mutableListOf<String>()
    // Private variables
    private var entriesMap = mutableMapOf<String, MutableList<entryDetails>>()
    private val excludedStrings = listOf("Apelyido", "Mga Pangalan", "Gitnang Apelyido", "Petsa ng Kapanganakan", "Tirahan",
        "Address", "First Name", "Last Name", "Middle Name", "Date of Birth", "Name", "Surname", "Given Names",)
    // Public functions
    fun getFromEntriesMap(key: String): MutableList<entryDetails>? {
        return entriesMap[key]
    }
    fun getEntryValue(key: String, entry: String): String? {
        /**
         * Example use:
         * val entry = getEntryValue("xxxx-xxxx-xxxx", "Date of Birth")
         * Note: xxxx-xxxx-xxxx is the identity number.
         */
        return entriesMap[key]?.find { it.typeOfEntry == entry }?.entry
    }
    fun processText(text: String) {
        /**
         * Processes the extracted text and adds it to the entriesMap.
         */
        if (determineStringType(text) == 1) {
            when (determineStringType(text)) {
                1 -> {
                    // Identity Number
                    if (text !in keyEntries) {
                        keyEntries.add(text)
                        entriesMap[text] = mutableListOf()
                    }
                }
                2 -> {
                    // Date of Birth
                    val key = keyEntries.last()
                    entriesMap[key]?.add(entryDetails("Date of Birth", text))
                }
                3 -> {
                    // Address
                    val key = keyEntries.last()
                    entriesMap[key]?.add(entryDetails("Address", text))
                }
                4 -> {
                    // Name
                    // TODO: Use Named Entity Recognition (NER) for extracting names information.
                    // TODO: Separate the last name, given name, and middle name.
                    val key = keyEntries.last()
                    entriesMap[key]?.add(entryDetails("Name", text))
                }
            }
        } else {
            Log.d("Unknown Type", "Unknown Type: $text")
        }

    }
    // Private helper functions
    private fun determineStringType(text: String): Int {
        /**
         * Returns the type of string extracted from the image.
         * 1. Identity Number
         * 2. Date of Birth
         * 3. Address
         * 4. Name (or Unknown Type)
         */
        if (text !in excludedStrings) {
            return if (isIdentityNumber(text)) {
                1
            } else if (isDateOfBirth(text)) {
                2
            } else if (isAddress(text)) {
                3
            } else {
                4
            }
        }
        return -1
    }
    private fun isAddress(text: String): Boolean {
        val regex = Regex("^[A-Z][a-z]{3,8}-[0-9]{1,2}-[0-9]{4}\$")
        return regex.matches(text)
    }
    private fun isIdentityNumber(text: String): Boolean {
        val regex = Regex("^[0-9]{4}-[0-9]{4}-[0-9]{4}\$")
        return regex.matches(text)
    }
    private fun isDateOfBirth(text: String): Boolean {
        val regex = Regex("^[A-Z][a-z]{3,8}-[0-9]{1,2}-[0-9]{4}\$")
        return regex.matches(text)
    }
}